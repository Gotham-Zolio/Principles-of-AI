"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

import random
from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        best_score = -float('inf')
        best_move = None
        
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            
            next_state = game_state.apply_move(move)
            score = -self.alphabeta(next_state, self.max_depth - 1, 
                                    -float('inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            best_move = Move.pass_turn()
        
        return best_move

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
        
        if maximizing_player:
            best_value = -float('inf')
            for move in game_state.legal_moves():
                if not move.is_play:
                    continue
                next_state = game_state.apply_move(move)
                value = self.minimax(next_state, depth - 1, False)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in game_state.legal_moves():
                if not move.is_play:
                    continue
                next_state = game_state.apply_move(move)
                value = self.minimax(next_state, depth - 1, True)
                best_value = min(best_value, value)
            return best_value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
        
        if maximizing_player:
            best_value = -float('inf')
            for move in game_state.legal_moves():
                if not move.is_play:
                    continue
                next_state = game_state.apply_move(move)
                value = self.alphabeta(next_state, depth - 1, 
                                       alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                
                if alpha >= beta:
                    break
            
            return best_value
        else:
            best_value = float('inf')
            for move in game_state.legal_moves():
                if not move.is_play:
                    continue
                next_state = game_state.apply_move(move)
                value = self.alphabeta(next_state, depth - 1, 
                                       alpha, beta, True)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                
                if beta <= alpha:
                    break
            
            return best_value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0  
            if winner.name == 'black':
                return 1.0 if game_state.next_player.name == 'white' else -1.0
            else:
                return 1.0 if game_state.next_player.name == 'black' else -1.0
        
        board = game_state.board
        black_stones = 0
        white_stones = 0
        black_liberties = 0
        white_liberties = 0
        
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                stone = board.get(Point(row, col))
                if stone == Player.black:
                    black_stones += 1
                elif stone == Player.white:
                    white_stones += 1
        
        processed = set()
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                if point in processed:
                    continue
                go_string = board.get_go_string(point)
                if go_string is not None:
                    processed.update(go_string.stones)
                    if go_string.color == Player.black:
                        black_liberties += go_string.num_liberties
                    else:
                        white_liberties += go_string.num_liberties
        
        black_score = black_stones + black_liberties * 0.5
        white_score = white_stones + white_liberties * 0.5
        
        if game_state.next_player == Player.black:
            return black_score - white_score
        else:
            return white_score - black_score

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        # TODO: 实现棋步排序
        # 提示：优先检查吃子、提子、连络等好棋
        moves = game_state.legal_moves()
        return moves  # 目前无序



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        # TODO: 实现缓存逻辑（考虑深度优先替换策略）
        pass
