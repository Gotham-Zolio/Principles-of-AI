import sys
import time
from dlgo.goboard import GameState
from dlgo.gotypes import Player
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

def play_game(black_agent, white_agent, board_size=5, max_moves=100):
    """进行单局自动对弈"""
    game = GameState.new_game(board_size)
    moves_count = 0
    
    while not game.is_over() and moves_count < max_moves:
        if game.next_player == Player.black:
            move = black_agent.select_move(game)
        else:
            move = white_agent.select_move(game)
        
        game = game.apply_move(move)
        moves_count += 1
        
    return game.winner()

def run_benchmark(black_name, black_agent_cls, white_name, white_agent_cls, num_games=10, **kwargs):
    """运行对局评测并收集结果"""
    print(f"[{black_name} (黑) vs {white_name} (白)] 进行 {num_games} 局对弈...")
    stats = {'black_wins': 0, 'white_wins': 0, 'draw_or_timeout': 0}
    
    for i in range(num_games):
        sys.stdout.write(f"\r  进度: {i+1}/{num_games}")
        sys.stdout.flush()
        
        # 针对不同类别的Agent进行不同的实例化
        black_agent = black_agent_cls(num_rounds=100) if black_name == "MCTS" else (black_agent_cls(max_depth=2) if black_name == "Minimax" else black_agent_cls())
        white_agent = white_agent_cls(num_rounds=100) if white_name == "MCTS" else (white_agent_cls(max_depth=2) if white_name == "Minimax" else white_agent_cls())
        
        winner = play_game(black_agent, white_agent)
        
        if winner == Player.black:
            stats['black_wins'] += 1
        elif winner == Player.white:
            stats['white_wins'] += 1
        else:
            stats['draw_or_timeout'] += 1
            
    print() # 换行
    
    # 计算胜率
    b_rate = (stats['black_wins'] / num_games) * 100
    w_rate = (stats['white_wins'] / num_games) * 100
    d_rate = (stats['draw_or_timeout'] / num_games) * 100
    
    print(f"  > 结果: {black_name} 胜: {b_rate:.1f}% | {white_name} 胜: {w_rate:.1f}% | 平局/超时: {d_rate:.1f}%\n")
    return stats

if __name__ == "__main__":
    print("=" * 60)
    print(" 围棋AI 性能跑分测试 (Benchmark)")
    print("=" * 60)
    
    # 考虑到终端实际运行100局时间很长，这里默认使用 10局/轮以加快测试速度
    # 如果您的电脑算力足够且需要和报告中的“100局”对齐，可将 N_GAMES 设为 50 或 100
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
    
    # 表头格式化输出
    print(f"{'-'*65}")
    print(f"{'黑方 (Black)':<15} | {'白方 (White)':<15} | {'黑方胜率':<8} | {'白方胜率':<8}")
    print(f"{'-'*65}")
    
    for b_name, b_cls, w_name, w_cls in matchups:
        black_agent = b_cls(num_rounds=100) if b_name == "MCTS" else (b_cls(max_depth=2) if b_name == "Minimax" else b_cls())
        white_agent = w_cls(num_rounds=100) if w_name == "MCTS" else (w_cls(max_depth=2) if w_name == "Minimax" else w_cls())
        
        # 统计
        stats = {'b': 0, 'w': 0, 'd': 0}
        for i in range(N_GAMES):
            sys.stdout.write(f"\r正在测试: {b_name} vs {w_name} ({i+1}/{N_GAMES})...")
            sys.stdout.flush()
            winner = play_game(black_agent, white_agent)
            if winner == Player.black: stats['b'] += 1
            elif winner == Player.white: stats['w'] += 1
            else: stats['d'] += 1
            
        b_rate = (stats['b'] / N_GAMES) * 100
        w_rate = (stats['w'] / N_GAMES) * 100
        
        sys.stdout.write("\r" + " "*50 + "\r") # 清除进度条
        print(f"{b_name:<15} | {w_name:<15} | {b_rate:>7.1f}% | {w_rate:>7.1f}%")

    print(f"{'-'*65}")
    print("测试完毕！")