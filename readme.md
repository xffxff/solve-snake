# Solve the snake game with reinforcement learning algorithms.

## Installation
### Create the python environment
```
conda create -n snake python=3.6
source activate snake
```

### Install [openai/baselines](https://github.com/openai/baselines.git)
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

### Install snake env
```
git clone https://github.com/XFFXFF/snake-gym.git
cd snake-gym
pip install -e .
```

### Install solve-snake
```
git clone https://github.com/XFFXFF/solve-snake.git
cd solve-snake
pip install -e .
```

## Running Tests
### Training a single agent model with ppo
```
python single_agent/ppo.py
```