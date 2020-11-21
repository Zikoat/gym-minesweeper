# gym-minesweeper
An OpenAI gym environment for MineSweeper

## Reward
The reward is the amount of cells opened. A perfect game gets a score of 1, and an opened mine subtracts 1 score, while pressing a cell that has no effect gives a small penalty. 

## Features
- [x] flooding
- [x] human interactivity
- [ ] medium and hard difficulty
- [ ] gridworld navigation and exploration (for large boards)
- [ ] chording (not planned)
- [ ] flagging (not planned)

## Installaton
```bash
git clone https://github.com/Zikoat/gym-minesweeper.git
cd gym-minesweeper
pip install -e .
```

## Usage

in python
```python
import gym
import gym_minesweeper
env = gym.make("Minesweeper-v0")
env.reset()
output = env.step(env.action_space.sample())
env.render()
```

`env.render()` creates an interactive matplotlib window where you can click with
your mouse to open cells. Alternatively, use `env.render("terminal")` to print
the board as ASCI:

```python
env.render("terminal")

. . . . . 1     
. . . . . 1     
. . . . 2 1     
. . . 1 1       
. . . 1         
. . . 1     1 1 
. . . 3 2 1 1 . 
. . . . . . . . 
```

the info returned by`env.step(action)` contains useful info, like the amount of opened cells, 

```python
output.info

{
    "opened cells": 29,
    "steps": 1,
    "unnecessary steps": 0,
    "game over": False,
    "died this turn": False,
    "opened cell": (0, 7),
    "mine locations": array([[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0]])
}
```
