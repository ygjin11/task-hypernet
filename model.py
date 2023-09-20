import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import torch.nn.functional as F
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from hypernetwork import HyperNetwork

# Set the device to CUDA device with index 1 (if available)
device = torch.device("cuda:1")

# Function to initialize model parameters
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Define Actor-Critic model class
class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.embedding_size = self.image_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Create an instance of HyperNetwork
        self.hope = HyperNetwork(z_dim=8)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, period, exp_memory_state, exp_memory_value, exp_memory_text):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x
        text = obs.text

        # Initialize tensors for otask1 and otask2
        otask1 = torch.eye(8, dtype=torch.float, device=device).requires_grad_()
        otask2 = torch.cat((torch.eye(7, dtype=torch.float, device=device), torch.ones(7, dtype=torch.float, device=device).unsqueeze(1)), dim=1).requires_grad_()

        # Load the 'task_lang.pt' file
        task = torch.load('task_lang.pt', map_location=device).float()

        # Initialize tensors for task1 and task2
        task1 = task[:7].to(device)
        task2 = torch.cat((task[:7], torch.ones(7).to(device).unsqueeze(1)), dim=1)

        embedding_new = []
        task_mapping = {
            1: (task1_1, otask1_1),
            24: (task1_2, otask1_2),
            5: (task1_3, otask1_3),
            9: (task1_4, otask1_4),
            17: (task1_5, otask1_5),
            15: (task1_6, otask1_6),
            26: (task1_7, otask1_7)
        }
        task = None
        otask = None
        for i in range(len(embedding)):
            text[i] = text[i].reshape(-1)
            item_value = text[i][0].item()
            if item_value in task_mapping:
                task, otask = task_mapping[item_value]

            matrix, bias = self.hope(task, otask)
            matrix = matrix.reshape(64, 64)
            bias = bias.reshape(64)
            embedding_new_i = torch.mm(embedding[i].reshape(1, -1), matrix) + bias
            embedding_new.append(embedding_new_i)
        embedding_hyper = torch.cat(embedding_new, dim=0)
        x = self.actor(embedding_hyper)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding_hyper)
        value = x.squeeze(1)

        sample_num = 50
        sample = random.sample(range(0, 50), sample_num)
        exp_memory_state = [exp_memory_state[i].to(device) for i in sample]
        exp_memory_value = [exp_memory_value[i].reshape(1).to(device) for i in sample]
        exp_memory_text = [exp_memory_text[i].to(device) for i in sample]
        exp_memory_state = torch.cat(exp_memory_state, dim=0)
        exp_memory_state = self.image_conv(exp_memory_state).reshape(exp_memory_state.shape[0], -1)

        query = []
        task_mapping = {
            1: (task2_1, otask2_1),
            24: (task2_2, otask2_2),
            5: (task2_3, otask2_3),
            9: (task2_4, otask2_4),
            17: (task2_5, otask2_5),
            15: (task2_6, otask2_6),
            26: (task2_7, otask2_7)
        }
        task = None
        otask = None
        for i in range(len(embedding)):
            item = text[i][0].item()
            if item in task_mapping:
                task, otask = task_mapping[item]

            matrix, bias = self.hope(task, otask)
            matrix = matrix.reshape(64, 64)
            bias = bias.reshape(64)
            query_i = torch.mm(embedding[i].reshape(1, -1), matrix) + bias
            query.append(query_i)
        query = torch.cat(query, dim=0)

        key = []
        task_mapping = {
            1: (task2_1, otask2_1),
            24: (task2_2, otask2_2),
            5: (task2_3, otask2_3),
            9: (task2_4, otask2_4),
            17: (task2_5, otask2_5),
            15: (task2_6, otask2_6),
            26: (task2_7, otask2_7)
        }
        task = None
        otask = None
        for i in range(sample_num):
            exp_memory_text[i] = exp_memory_text[i].reshape(-1)
            item = exp_memory_text[i][0].item()
            if item in task_mapping:
                task, otask = task_mapping[item]

            matrix, bias = self.hope(task, otask)
            matrix = matrix.reshape(64, 64)
            bias = bias.reshape(64)
            key_i = torch.mm(exp_memory_state[i].reshape(1, -1), matrix) + bias
            key.append(key_i)
        key = torch.cat(key, dim=0)

        # Calculate attention weights
        attention_weights = torch.mm(query, key.T)
        attention_weights = F.softmax(attention_weights, dim=1)

        lam = []
        for idx in range(len(attention_weights)):
            lam.append(torch.max(attention_weights[idx]).reshape(1, -1))
        lam = torch.cat(lam).reshape(-1)

        exp_memory_value = torch.cat(exp_memory_value)
        exp_memory_value = exp_memory_value.reshape(1, -1)
        memory_value = torch.mm(attention_weights, exp_memory_value.T).reshape(-1)

        return dist, value, memory_value, lam, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
