# Principles-of-AI

这是“人工智能原理”课程的编程作业仓库。

## HW1: 围棋 AI (Go AI)

本项目实现了一个在 $5\times 5$ 棋盘上运行的围棋（Go）对弈程序，包含底层规则引擎、多种智能对弈算法以及可视化的人机交互界面。

### 🌟 核心特性 (Features)
* **规则引擎**：实现了完整的围棋落子逻辑，包括吃子（提子）和禁着点（自杀）判断。
* **随机智能体 (Random Agent)**：在当前所有的合法落子点中随机选择，作为基线对比模型。
* **蒙特卡洛树搜索 (MCTS Agent)**：基于UCT公式，并结合了走子启发式规则与模拟深度限制进行优化，具备较高的对弈胜率。
* **极小极大搜索 (Minimax Agent)**：基于深度优先搜索，结合 Alpha-Beta 剪枝，配合盘面气数差启发式评估函数，在小棋盘下表现出色。
* **图形化界面 (GUI)**：基于 `tkinter` 库开发的可视化界面，使用多线程（Daemon Thread）保证在AI深度计算时界面不卡死，支持玩家与任意 AI 进行对战。
* **性能评测 (Benchmark)**：提供了完善的终端批量对局脚本，可自动统计各 AI 之间的胜率和计算耗时。

### 🚀 快速开始 (Quick Start)

请先进入 `HW1` 目录：
```bash
cd HW1
```

#### 1. 启动图形化界面 (GUI) 与 AI 对弈
你可以直接运行 GUI 程序，并在启动时通过命令行参数指定你想对战的 AI 对手：
```bash
python gui.py --agent mcts     # 与 MCTS AI 对战 (默认)
python gui.py --agent minimax  # 与 Minimax AI 对战
python gui.py --agent random   # 与 Random AI 对战
```

#### 2. 终端自动化测试 (CLI)
使用 `play.py` 可以让两个 AI 在终端内部直接进行对局（不支持人类输入）：
```bash
python play.py --agent1 mcts --agent2 minimax --size 5 --games 10 --quiet
```
*参数说明：*
* `--agent1` / `--agent2`: 指定黑白双方的 AI 类型 (`mcts`, `minimax`, `random`)。
* `--size`: 棋盘大小，默认为 5。
* `--games`: 连续自动对弈的总局数。
* `--quiet`: 静默模式，只在最后输出胜率与耗时统计表格，不打印对弈中间过程。

#### 3. 运行评测跑分 (Benchmark)
一键运行内置的多组对战测试并生成包含胜率、耗时的性能跑分表：
```bash
python benchmark.py
```

### 📁 目录结构
```text
HW1/
├── agents/             # 各类 AI 算法的实现
│   ├── __init__.py
│   ├── mcts_agent.py   # MCTS 实现
│   ├── minimax_agent.py# Minimax + Alpha-Beta 剪枝
│   └── random_agent.py # Random 实现
├── dlgo/               # 围棋底层游戏引擎 (包含状态管理与落子规则)
├── docs/               # 文档等其他静态资源
├── report/             # LaTeX 实验报告源码
├── benchmark.py        # 跑分评测测试脚本
├── gui.py              # Tkinter 图像化游戏界面入口
├── play.py             # 终端 CLI 自动化自动对战入口
└── README.md           # HW1 子项目说明
```