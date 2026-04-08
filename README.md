# Principles of AI

This is the programming assignment repository for the "Principles of Artificial Intelligence" course.

## HW1: Go AI

This project implements a Go (Weiqi) gameplay program that runs on a $5\times 5$ board. It includes a complete underlying rules engine, several intelligent gameplay algorithms, and a visual human-computer interaction interface.

### 🌟 Core Features
* **Rules Engine**: Implements the complete Go logic, including capturing stones and detecting illegal moves (suicide).
* **Random Agent**: Randomly selects a move from all currently legal options. Served as a baseline model.
* **Monte Carlo Tree Search (MCTS Agent)**: Based on the UCT formula, optimized with heuristic move rules and simulation depth limits, exhibiting a reasonably high win rate.
* **Minimax Search (Minimax Agent)**: Based on depth-first search combined with Alpha-Beta pruning, accompanied by a heuristic evaluation function based on the liberty difference on the board. Performs excellently on small boards.
* **Graphical User Interface (GUI)**: A visual interface built with `tkinter`. It uses Daemon Threads to ensure the UI remains responsive during deep AI computations and supports users playing against any AI algorithm.
* **Performance Benchmark**: Provides a comprehensive terminal batch script to automatically calculate the win rate and computation time among different AIs.

### 🚀 Quick Start

First, navigate to the `HW1` directory:
```bash
cd HW1
```

#### 1. Launch GUI and Play Against AI
You can directly run the GUI program and specify your AI opponent via the command-line argument at startup:
```bash
python gui.py --agent mcts     # Play against MCTS AI (Default)
python gui.py --agent minimax  # Play against Minimax AI
python gui.py --agent random   # Play against Random AI
```

#### 2. Terminal Automated CLI Match
Use `play.py` to allow two AIs to play directly in the terminal (human input is not supported here):
```bash
python play.py --agent1 mcts --agent2 minimax --size 5 --games 10 --quiet
```
*Arguments:*
* `--agent1` / `--agent2`: Specify the AI type for Black and White (`mcts`, `minimax`, `random`).
* `--size`: Size of the board. The default is 5.
* `--games`: Total number of continuous auto-play matches.
* `--quiet`: Quiet mode. Only outputs the final win rate and time statistics table without printing the intermediate board states.

#### 3. Run Benchmarks
Run a built-in multiple match test with one click to generate performance benchmarks containing win rates and processing times:
```bash
python benchmark.py
```

### 📁 Directory Structure
```text
HW1/
├── agents/             # Implementations of various AI algorithms
│   ├── __init__.py
│   ├── mcts_agent.py   # MCTS implementation
│   ├── minimax_agent.py# Minimax + Alpha-Beta pruning
│   └── random_agent.py # Random baseline implementation
├── dlgo/               # Go underlying engine (game state & rules)
├── docs/               # Documentations and other static resources
├── report/             # LaTeX report source files
├── benchmark.py        # Benchmark testing script
├── gui.py              # Tkinter graphical interface entry point
├── play.py             # CLI automated match entry point
└── README.md           # Instructions for HW1 subsystem
```