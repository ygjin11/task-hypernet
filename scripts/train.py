import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
import torch
import utils
from utils import device
from model import ACModel
import random

device = torch.device("cuda:1")

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
# parser.add_argument("--env", required=True,
#                     help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    # default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)
    

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    env_name = ["MiniGrid-DoorKey-6x6-v0", "MiniGrid-DistShift1-v0",
                           "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-LavaGapS7-v0",
                           "MiniGrid-MemoryS11-v0", "MiniGrid-SimpleCrossingS9N2-v0", "MiniGrid-MultiRoom-N2-S4-v0"]

    env_num = len(env_name)
    index = 0
    for i in range(args.procs):
        envs.append(utils.make_env(env_name[index], args.seed + 10000 * i))
        print(env_name[index])
        index = index + 1
        if index == env_num:
            index = 0
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    # load memory_state&value
    exp_memory_state = torch.load('memorystate.pt', map_location=device)[0:50]
    exp_memory_value = torch.load('memory/value.pt', map_location=device)[0:50]
    exp_memory_text = torch.load('memory/text.pt', map_location=device)[0:50]
    # exp_memory_value.shape:100.1.3.7.7
    # exp_memory_state.shape:100.64

    # exp_memory_size
    exp_memory_size = len(exp_memory_state)
    # default update size
    update_size_default = exp_memory_size
    # stage of training
    stage = 0
    while num_frames < args.frames:
        period = num_frames / args.frames
        # Update model parameters
        update_start_time = time.time()
        # exps
        exps, logs1, new_exp_memory = algo.collect_experiences(period=period, exp_memory_value=exp_memory_value, exp_memory_state=exp_memory_state, exp_memory_text=exp_memory_text)

        # value&reward preprocess
        reward_norm = []
        value_norm = []
        for i in range(len(new_exp_memory)):
            reward_norm.append(new_exp_memory[i][2].item())
            value_norm.append(new_exp_memory[i][1].item())
        reward_norm = torch.tensor(reward_norm)
        value_norm = torch.tensor(value_norm)
        reward_norm_mean = torch.mean(reward_norm)
        reward_norm_var = torch.var(reward_norm, unbiased=False)
        value_norm_mean = torch.mean(value_norm)
        value_norm_var = torch.var(value_norm, unbiased=False)
        for i in range(len(new_exp_memory)):
            new_exp_memory[i][2] = reward_norm[i] - reward_norm_mean + value_norm_mean

        # update exp_memory
        if len(new_exp_memory) < update_size_default:
            update_size = len(new_exp_memory)
        else:
            update_size = update_size_default
        index_random = [random.randint(0, len(new_exp_memory)-1) for i in range(update_size)]
        for index in range(update_size):
            exp_memory_state[stage] = new_exp_memory[index_random[index]][0].image.transpose(1, 3).transpose(2, 3)
            exp_memory_value[stage] = new_exp_memory[index_random[index]][2]
            exp_memory_text[stage] = new_exp_memory[index_random[index]][0].text
            stage = stage + 1
            if stage == exp_memory_size:
                stage = 0

        # update
        logs2 = algo.update_parameters(exps, period=period, exp_memory_state=exp_memory_state, exp_memory_value=exp_memory_value, exp_memory_text=exp_memory_text)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"]]

            txt_logger.info(
                # "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
