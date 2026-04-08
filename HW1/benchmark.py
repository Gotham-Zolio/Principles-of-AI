import sys
import time
from dlgo.gotypes import Player
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

# 直接导入 play.py 脚本中的核心对局驱动方法
from play import play_game

def pad(text, width):
    """辅助函数：自动根据中文字符调整占位宽度进行对齐填充"""
    # 中文字符（非ASCII）按 2 的跨度计算宽度
    display_len = sum(2 if ord(c) > 127 else 1 for c in str(text))
    return str(text) + " " * max(0, width - display_len)

class TimeWrapper:
    """代理器封装层：通过该封装将 Agent 传入并拦截计算耗时"""
    def __init__(self, agent):
        self.agent = agent
        self.total_time = 0.0
        self.moves = 0
        
    def __call__(self, game):
        start = time.time()
        move = self.agent.select_move(game)
        self.total_time += time.time() - start
        self.moves += 1
        return move

if __name__ == "__main__":
    print("=" * 95)
    print(" 围棋AI 性能跑分测试 (Benchmark - 基于 play.py 框架调用)")
    print("=" * 95)
    
    # 每组进行10局对测试
    N_GAMES = 10 
    
    matchups = [
        ("MCTS", MCTSAgent, "Random", RandomAgent),
        ("Random", RandomAgent, "MCTS", MCTSAgent),
        ("Minimax", MinimaxAgent, "Random", RandomAgent),
        ("Random", RandomAgent, "Minimax", MinimaxAgent),
        ("MCTS", MCTSAgent, "Minimax", MinimaxAgent),
        ("Minimax", MinimaxAgent, "MCTS", MCTSAgent),
    ]
    
    print(f"每组对抗进行 {N_GAMES} 局测试 (棋盘: 5x5)\n")
    
    # --------- 规整的表头 ---------
    head_b = pad("黑方(Black)", 16)
    head_w = pad("白方(White)", 16)
    head_b_win = pad("黑方胜率", 10)
    head_w_win = pad("白方胜率", 10)
    head_b_time = pad("黑平均耗时/步", 15)
    head_w_time = pad("白平均耗时/步", 15)
    
    table_width = 95
    print("-" * table_width)
    print(f"{head_b} | {head_w} | {head_b_win} | {head_w_win} | {head_b_time} | {head_w_time}")
    print("-" * table_width)
    
    for b_name, b_cls, w_name, w_cls in matchups:
        stats = {'b': 0, 'w': 0, 'd': 0}
        b_total_time, b_total_moves = 0.0, 0
        w_total_time, w_total_moves = 0.0, 0
        
        for i in range(N_GAMES):
            sys.stdout.write(f"\r正在测试: {b_name} vs {w_name} ({i+1}/{N_GAMES})...")
            sys.stdout.flush()
            
            # 使用各 Agent类 重新生成全新实例防污染，MCTS传入num_rounds，Minimax传入max_depth
            black_agent = b_cls(num_rounds=100) if b_name == "MCTS" else (b_cls(max_depth=2) if b_name == "Minimax" else b_cls())
            white_agent = w_cls(num_rounds=100) if w_name == "MCTS" else (w_cls(max_depth=2) if w_name == "Minimax" else w_cls())
            
            # 使用装时封装层包裹
            bw = TimeWrapper(black_agent)
            ww = TimeWrapper(white_agent)
            
            # 这里【修改】为直接调取 play.py 的测试逻辑机制 (verbose=False表示静默模式)
            winner, _, _ = play_game(bw, ww, board_size=5, verbose=False)
            
            if winner == Player.black: stats['b'] += 1
            elif winner == Player.white: stats['w'] += 1
            else: stats['d'] += 1
            
            b_total_time += bw.total_time
            b_total_moves += bw.moves
            w_total_time += ww.total_time
            w_total_moves += ww.moves
            
        b_rate = f"{(stats['b'] / N_GAMES) * 100:.1f}%"
        w_rate = f"{(stats['w'] / N_GAMES) * 100:.1f}%"
        
        b_avg_time = f"{b_total_time / max(1, b_total_moves):.4f}s"
        w_avg_time = f"{w_total_time / max(1, w_total_moves):.4f}s"
        
        # 使用pad统一对齐占位输出表格
        sys.stdout.write("\r" + " "*60 + "\r") # 清除进度条
        print(f"{pad(b_name, 16)} | {pad(w_name, 16)} | {pad(b_rate, 10)} | {pad(w_rate, 10)} | {pad(b_avg_time, 15)} | {pad(w_avg_time, 15)}")

    print("-" * table_width)
    print("测试完毕！您可对上方带用时的统列表进行截图保存。")