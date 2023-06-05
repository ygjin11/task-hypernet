# DRL-with-Multitask-EM-Based-on-Task-Conditioned-Hypernetwork

## Model
[![]("model.png")](https://github.com/ygjin11/DRL-with-Multitask-EM-Based-on-Task-Conditioned-Hypernetwork/blob/main/model.png)

## Installation

1. Clone this repository.

2. conda create -n minigrid python=3.9

3.  Install requirements.
```
pip3 install -r requirements.txt
```


## Example of use

Train an agent.
```
python3 -m scripts.train --algo ppo --model model --save-interval 10 --frames 80000
```

