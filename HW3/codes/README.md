### *Pac-Man 强化学习* - 使用文档

1) 运行值迭代（DP），查看策略与 V 值：
   
   ```
   python Pac-Man.py --mode dp
   ```

2) 运行蒙特卡洛训练（MC），并生成学习曲线：
   
   ```
   python Pac-Man.py --mode mc --episodes 3000
   ```

3) 在 GUI 中实时演示 MC 学习过程：
   
   ```
   python Pac-Man.py --mode learn --live-episodes 1000
   ```

- 脚本会在当前目录或 --save-dir 指定目录生成 dp_policy.png, mc_policy.png, learning_curves.png 等文件。