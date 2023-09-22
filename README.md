# DRL-with-Multitask-EM-Based-on-Task-Conditioned-Hypernetwork

Code of paper "Deep Reinforcement Learning with Multitask Episodic Memory Based on Task-Conditioned Hypernetwork"(https://arxiv.org/abs/2306.10698).

## Model

<img src="https://github.com/ygjin11/DRL-with-Multitask-EM-Based-on-Task-Conditioned-Hypernetwork/blob/main/modelarc.png" width="600px" />

## Installation

1. Clone this repository.

2. Install requirements.
```
cd task-hypernet
conda create -n minigrid python=3.9
pip install -r requirements.txt
```

## Example of use

Train an agent.
```
python3 -m scripts.train --algo ppo --model model --save-interval 10 --frames 80000
```

