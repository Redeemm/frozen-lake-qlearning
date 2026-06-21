# Frozen Lake Q-Learning

DSCD 614 – Reinforcement Learning · Assignment 1

## Setup

```bash
pip3 install -r requirements.txt
```

## Run

```bash
# Train the agent (50,000 episodes → saves Q-table + training curves)
python3 train.py

# Evaluate the trained policy (500 episodes, greedy)
python3 evaluate.py

# Generate PDF report
python3 generate_report.py
```

## Results

| Metric         | Value    |
|----------------|----------|
| Success Rate   | 100%     |
| Avg Reward     | 1.0      |
| Episodes       | 500/500  |

## Structure

```
├── environment.py      # FrozenLakeEnv
├── agent.py            # QLearningAgent
├── train.py            # Training loop + plots
├── evaluate.py         # Evaluation
├── generate_report.py  # PDF report
├── results/
│   ├── q_table.npy
│   ├── train_stats.json
│   ├── eval_results.json
│   └── training_curves.png
```
