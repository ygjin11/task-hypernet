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
device = torch.device("cuda:1")


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


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

        # self.num = 50
        # self.hope = HyperNetwork(z_dim=512+self.num)

        self.hope = HyperNetwork(z_dim=8)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, period, exp_memory_state, exp_memory_value, exp_memory_text):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x
        text = obs.text

        otask1_1 = torch.nn.Parameter(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).float().to(device), requires_grad=True)
        otask1_2 = torch.nn.Parameter(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]).float().to(device), requires_grad=True)
        otask1_3 = torch.nn.Parameter(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]).float().to(device), requires_grad=True)
        otask1_4 = torch.nn.Parameter(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]).float().to(device), requires_grad=True)
        otask1_5 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]).float().to(device), requires_grad=True)
        otask1_6 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]).float().to(device), requires_grad=True)
        otask1_7 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 0, 1 ,0]).float().to(device), requires_grad=True)

        otask2_1 = torch.nn.Parameter(torch.tensor([1, 0, 0, 0, 0, 0, 0, 1]).float().to(device), requires_grad=True)
        otask2_2 = torch.nn.Parameter(torch.tensor([0, 1, 0, 0, 0, 0, 0, 1]).float().to(device), requires_grad=True)
        otask2_3 = torch.nn.Parameter(torch.tensor([0, 0, 1, 0, 0, 0, 0, 1]).float().to(device), requires_grad=True)
        otask2_4 = torch.nn.Parameter(torch.tensor([0, 0, 0, 1, 0, 0, 0, 1]).float().to(device), requires_grad=True)
        otask2_5 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 1, 0, 0, 1]).float().to(device), requires_grad=True)
        otask2_6 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 1, 0, 1]).float().to(device), requires_grad=True)
        otask2_7 = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]).float().to(device), requires_grad=True)


        task = torch.load('task_lang.pt')

        task1_1 = task[0].float().to(device)
        task1_2 = task[1].float().to(device)
        task1_3 = task[2].float().to(device)
        task1_4 = task[3].float().to(device)
        task1_5 = task[4].float().to(device)
        task1_6 = task[5].float().to(device)
        task1_7 = task[6].float().to(device)

        task2_1 = task[0].float().to(device)
        task2_2 = task[1].float().to(device)
        task2_3 = task[2].float().to(device)
        task2_4 = task[3].float().to(device)
        task2_5 = task[4].float().to(device)
        task2_6 = task[5].float().to(device)
        task2_7 = task[6].float().to(device)


        embedding_new = []
        for i in range(len(embedding)):
            text[i] = text[i].reshape(-1)
            if text[i][0].item() == 1:
                task = task1_1
                otask = otask1_1
            if text[i][0].item() == 24:
                task = task1_2
                otask = otask1_2
            if text[i][0].item() == 5:
                task = task1_3
                otask = otask1_3
            if text[i][0].item() == 9:
                task = task1_4
                otask = otask1_4
            if text[i][0].item() == 17:
                task = task1_5
                otask = otask1_5
            if text[i][0].item() == 15:
                task = task1_6
                otask = otask1_6
            if text[i][0].item() == 26:
                task = task1_7
                otask = otask1_7

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

        sample_num = 15
        sample = random.sample(range(0, 50), sample_num)
        exp_memory_state = [exp_memory_state[i].to(device) for i in sample]
        exp_memory_value = [exp_memory_value[i].reshape(1).to(device) for i in sample]
        exp_memory_text = [exp_memory_text[i].to(device) for i in sample]
        exp_memory_state = torch.cat(exp_memory_state, dim=0)
        exp_memory_state = self.image_conv(exp_memory_state).reshape(exp_memory_state.shape[0], -1)

        query = []
        for i in range(len(embedding)):
            if text[i][0].item() == 1:
                task = task2_1
                otask = otask2_1
            if text[i][0].item() == 24:
                task = task2_2
                otask = otask2_2
            if text[i][0].item() == 5:
                task = task2_3
                otask = otask2_3
            if text[i][0].item() == 9:
                task = task2_4
                otask = otask2_4
            if text[i][0].item() == 17:
                task = task2_5
                otask = otask2_5
            if text[i][0].item() == 15:
                task = task2_6
                otask = otask2_6
            if text[i][0].item() == 26:
                task = task2_7
                otask = otask2_7
            matrix, bias = self.hope(task, otask)
            matrix = matrix.reshape(64, 64)
            bias = bias.reshape(64)
            query_i = torch.mm(embedding[i].reshape(1, -1), matrix) + bias
            query.append(query_i)
        query = torch.cat(query, dim=0)

        key = []
        for i in range(sample_num):
            exp_memory_text[i] = exp_memory_text[i].reshape(-1)
            if exp_memory_text[i][0].item() == 1:
                task = task2_1
                otask = otask2_1
            if exp_memory_text[i][0].item() == 24:
                task = task2_2
                otask = otask2_2
            if exp_memory_text[i][0].item() == 5:
                task = task2_3
                otask = otask2_3
            if exp_memory_text[i][0].item() == 9:
                task = task2_4
                otask = otask2_4
            if exp_memory_text[i][0].item() == 17:
                task = task2_5
                otask = otask2_5
            if exp_memory_text[i][0].item() == 15:
                task = task2_6
                otask = otask2_6
            if exp_memory_text[i][0].item() == 26:
                task = task2_7
                otask = otask2_7
            matrix, bias = self.hope(task, otask)
            matrix = matrix.reshape(64, 64)
            bias = bias.reshape(64)
            key_i = torch.mm(exp_memory_state[i].reshape(1, -1), matrix) + bias
            key.append(key_i)
        key = torch.cat(key, dim=0)

        # query * key
        # attention_weights.shape(16,100)
        # attention_weights = torch.mm(embedding_hyper, key.T)
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


