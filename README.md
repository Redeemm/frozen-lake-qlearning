# Frozen Lake Q-Learning from First Principles



## Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a paradigm in which an autonomous agent learns decision-making by interacting with an environment. At each time step the agent observes the current state, selects an action, and receives a reward signal together with a transition to a new state. The objective is to discover a policy that maximises the expected sum of discounted future rewards. Unlike supervised learning, no labelled training data is provided; feedback is sparse and often delayed.

### What is Frozen Lake?

Frozen Lake is an 8×8 grid-world problem. The agent must navigate from the start cell (S) to the goal cell (G) without falling into any hole (H). The map used in this assignment is:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

- **S** — Start (position 0)
- **F** — Frozen, safe
- **H** — Hole, terminal (episode ends)
- **G** — Goal, terminal (reward +1)

---

## Environment Design

Implemented in `environment.py` as the `FrozenLakeEnv` class, with no external RL frameworks.

### State Representation

Each of the 64 grid cells is encoded as `state = row × 8 + col`, giving a flat state space of size 64.

### Action Representation

| Index | Direction | Row delta | Col delta |
|-------|-----------|-----------|-----------|
| 0 | Left | 0 | −1 |
| 1 | Down | +1 | 0 |
| 2 | Right | 0 | +1 |
| 3 | Up | −1 | 0 |

Movement at grid boundaries is clamped — the agent stays in place.

### Reward Structure

| Event | Reward |
|-------|--------|
| Reach Goal (G) | +1.0 |
| Fall in Hole (H) | −1.0 |
| Safe frozen step | 0.0 |

A negative hole reward is essential: purely sparse reward (+1 at goal only) makes learning prohibitively slow because random exploration rarely reaches G. The hole penalty provides immediate negative signals that propagate hole-avoidance information through the Q-table via the Bellman update.

---

## Q-Learning Algorithm

Implemented in `agent.py` as the `QLearningAgent` class.

### Description

Q-Learning is a **model-free, off-policy** temporal-difference algorithm. It maintains a Q-table `Q[s, a]` estimating the expected discounted return for each (state, action) pair, initialised to zero.

### Update Equation

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

- `α` — learning rate: controls how quickly estimates are revised
- `γ` — discount factor: weights future rewards relative to immediate rewards
- `r` — reward received
- `max_a' Q(s', a')` — bootstrapped target from the next state

When an episode ends, the bootstrap term collapses to zero and the target reduces to `r` alone.

### Exploration Strategy

**Epsilon-greedy with multiplicative decay:**

- With probability `ε`: choose a random action (explore)
- With probability `1 − ε`: choose `argmax_a Q(s, a)` (exploit)
- After each episode: `ε ← max(ε_min, ε × decay_rate)`

This ensures broad exploration early in training, transitioning smoothly to near-greedy exploitation as the Q-table converges.

---

## Training Procedure

Implemented in `train.py`.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate α | 0.1 | Balances speed and stability |
| Discount factor γ | 0.99 | High value; long-horizon planning |
| Initial ε | 1.0 | Full exploration at start |
| ε decay rate | 0.9999 | Slow decay; exploration active for ~23k eps |
| ε minimum | 0.01 | Retains 1% random exploration at convergence |
| Episodes | 50,000 | Sufficient for full convergence on 8×8 map |
| Max steps / episode | 200 | Prevents infinite loops in early training |

### Number of Episodes

50,000 episodes were used. A decay of 0.9995 caused epsilon to collapse before the agent had seen enough goal transitions; 0.9999 kept epsilon above 0.10 for the first ~23,000 episodes, giving the agent ample time to explore.

---

## Results

### Final Success Rate

| Metric | Value |
|--------|-------|
| Episodes evaluated | 500 |
| Successful runs | 500 |
| Failures | 0 |
| Success Rate | **100%** |
| Average Reward | 1.000000 |

Training convergence by window:

| Episode Range | Success Rate |
|---------------|-------------|
| 0 – 5,000 | 12.3% |
| 5,001 – 10,000 | 31.9% |
| 10,001 – 15,000 | 53.9% |
| 20,001 – 25,000 | 82.8% |
| 30,001 – 35,000 | 93.3% |
| 45,001 – 50,000 | **98.5%** |

### Learned Policy

```
+----+----+----+----+----+----+----+----+
|  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |
|  → |  → |  → |  → |  ↓ |  ↓ |  ↓ |  ↓ |
|  → |  → |  ↑ |  H |  ↓ |  ↓ |  ↓ |  ↓ |
|  ↑ |  ↑ |  ↑ |  H |  ↓ |  ↓ |  ↓ |  ↓ |
|  ↑ |  ↑ |  ← |  H |  → |  → |  → |  ↓ |
|  ↑ |  H |  H |  → |  ↑ |  ↑ |  H |  ↓ |
|  ↑ |  H |  ↓ |  ← |  H |  ↑ |  H |  ↓ |
|  ↑ |  ← |  ← |  H |  ↓ |  ↑ |  → |  G |
+----+----+----+----+----+----+----+----+
```

### Discussion

The policy reflects two main routes to the goal: (1) a right-then-down path through columns 4–7 that avoids the column-3 hole barrier, and (2) a downward sweep along the right edge. The training curves (`results/training_curves.png`) show smooth, monotonic improvement. The exploration schedule (ε-decay = 0.9999) kept exploration active long enough for the agent to discover the goal frequently and fill in meaningful Q-values for safe paths. After convergence, the greedy policy achieves a 100% success rate.

---

## Execution Instructions

### Requirements

```bash
pip3 install -r requirements.txt
```

### Run Training

```bash
python3 train.py
```

Trains for 50,000 episodes, saves Q-table to `results/q_table.npy`, and writes training curves to `results/training_curves.png`.

### Run Evaluation

```bash
python3 evaluate.py
```

Loads the saved Q-table and evaluates the greedy policy over 500 episodes. Results saved to `results/eval_results.json`.

### Generate Report

```bash
python3 generate_report.py
```

Produces `report.pdf`.

---

## Repository Structure

```
frozen-lake-qlearning/
├── environment.py      # FrozenLakeEnv class
├── agent.py            # QLearningAgent class
├── train.py            # Training loop + plots
├── evaluate.py         # Evaluation
├── generate_report.py  # PDF report generation
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── q_table.npy
    ├── train_stats.json
    ├── eval_results.json
    └── training_curves.png
```
