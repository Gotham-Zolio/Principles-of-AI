Pac-Man.py 使用说明（HW3 / 精简版 GUI+训练脚本）

概述
- Pac-Man.py 实现了本次 HW3 作业要求的内容：状态建模、值迭代（DP）、First-Visit 蒙特卡洛控制（MC）、以及可选的 tkinter GUI 演示与实时学习（--mode learn）。

运行环境
- Python 3.8+，依赖：numpy, matplotlib, pillow
  安装：pip install -r requirements.txt （若无 requirements.txt，请手动安装 numpy matplotlib pillow）
- Windows 上建议设置：
  set PYTHONUTF8=1    # 防止中文路径/字体问题
  set MPLBACKEND=Agg  # 批量生成图片时避免阻塞（可选）

主要命令
- 值迭代（DP）：
  python Pac-Man.py --mode dp

- 蒙特卡洛训练（MC）：
  python Pac-Man.py --mode mc --episodes 3000

- 同时运行 DP 与 MC，并生成图像/动画：
  python Pac-Man.py --mode both --episodes 3000

- GUI 实时学习演示（观察 agent 从零开始学习）：
  python Pac-Man.py --mode learn --live-episodes 1000
  在 GUI 中可通过窗口下方的速度按钮切换步速（慢/正常/快速/极速），便于录制。

常用参数
- --episodes N       : MC 训练回合数（默认 3000）
- --live-episodes N  : learn 模式下的回合数（默认 1000）
- --gamma FLOAT      : 折扣因子（默认 0.99）
- --save-dir PATH    : 指定输出图片/动画保存目录

录制建议
- 录制学习过程时，开始用“慢速”展示早期探索（让观众看到 ε 较大时的随机性），后期可切“极速”加速到收敛阶段。
- 若需导出动画/MP4，确保系统已安装 ffmpeg；若缺失，脚本会自动降级为 GIF 保存。

文件说明（位于 HW3/codes）
- Pac-Man.py          : 主脚本（包含算法与 GUI）
- Pac-Man.slim.py     : 精简无 GUI 的实现（仅供快速测试/自动评测）
- 生成的图片与动画：dp_policy.png, mc_policy.png, learning_curves.png, Pac-Man-learning.mp4/gif

调试提示
- 若中文显示为方块，检查 matplotlib 字体设置（脚本在 import 时尝试加载常见中文字体）；Windows 上常用 Microsoft YaHei。
- 若动画无法生成 MP4，检查是否安装 ffmpeg，或查看脚本是否已降级为 GIF。