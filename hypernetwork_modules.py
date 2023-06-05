import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def init_linear_layer(linear_layer, std=1e-2):

    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)


class HyperNetwork(nn.Module):

    def __init__(self, z_dim=16, out_size=64*64, out_size_2=64):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.out_size = out_size
        self.out_size_2 = out_size_2
        self.hidden_size_1 = 64
        self.hidden_size_2 = 64

        self.task_embeding_generator = nn.Sequential(
            linear_layer(z_dim, int(self.hidden_size_1/2)),
            nn.ReLU(),
            linear_layer(int(self.hidden_size_1/2), self.hidden_size_1))

        self.task_embeding_generator_2 = nn.Sequential(
            linear_layer(512, self.hidden_size_2))

        self.weight_generator = nn.Sequential(
            linear_layer(self.hidden_size_1+self.hidden_size_2, self.out_size))
        self.bias_generator = nn.Sequential(
            linear_layer(self.hidden_size_1+self.hidden_size_2, self.out_size_2))

    def forward(self, task, o_task):
        otask = self.task_embeding_generator(o_task).view(-1)
        task = self.task_embeding_generator_2(task).view(-1)
        atask = torch.cat((task, otask), 0)
        weight = self.weight_generator(atask)
        bias = self.bias_generator(atask)
        return weight, bias

