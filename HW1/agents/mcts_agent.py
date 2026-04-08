"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

import random
import math
import time
from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MCTSAgent"]



class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move 
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        if not self.children:
            return None
        
        parent_log = math.log(self.visit_count)
        
        best_child = None
        best_uct = -float('inf')
        
        for child in self.children:
            if child.visit_count == 0:
                return child
            
            exploitation = child.value
            exploration = c * math.sqrt(parent_log / child.visit_count)
            uct = exploitation + exploration
            
            if uct > best_uct:
                best_uct = uct
                best_child = child
        
        return best_child

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        moves = self.game_state.legal_moves()
        
        for move in moves:
            if move.is_resign:
                continue
                
            new_game_state = self.game_state.apply_move(move)
            child = MCTSNode(
                new_game_state, 
                parent=self, 
                move=move,  
                prior=1.0
            )
            self.children.append(child)
        
        return random.choice(self.children) if self.children else None

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            opponent_value = 1.0 - value
            self.parent.backup(opponent_value)


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(self, num_rounds=1000, temperature=1.0):
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        root = MCTSNode(game_state)
        
        for _ in range(self.num_rounds):
            node = root
            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child(c=1.414)
            
            if not node.is_terminal():
                new_child = node.expand()
                if new_child is not None:
                    node = new_child
            
            result = self._simulate(node.game_state)
            
            node.backup(1.0 - result)
        
        return self._select_best_move(root)

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        if not root.children:
            return random.choice(root.game_state.legal_moves())
        
        best_child = max(root.children, key=lambda child: child.visit_count)
        
        if best_child.move is not None:
            return best_child.move
        
        return random.choice(root.game_state.legal_moves())

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        max_simulation_depth = 30
        current_state = game_state
        
        for _ in range(max_simulation_depth):
            if current_state.is_over():
                break
            
            move = self._heuristic_move(current_state)
            current_state = current_state.apply_move(move)
        
        if current_state.is_over():
            winner = current_state.winner()
            if winner == game_state.next_player:
                return 1.0  
            elif winner is None:
                return 0.5  
            else:
                return 0.0  
        else:
            return self._evaluate_position(current_state, game_state.next_player)

    def _heuristic_move(self, game_state):
        """
        启发式选择棋步（优化策略1）。

        优先选择：
        1. 已有棋子邻近的位置
        2. 正中央位置（对小棋盘更优）
        3. 能提子的位置
        4. 随机落子或 pass

        Args:
            game_state: 当前游戏状态

        Returns:
            选中的合法棋步
        """
        candidates = game_state.legal_moves()
        
        play_moves = [m for m in candidates if m.is_play]
        
        if not play_moves:
            return random.choice(candidates)
        
        weighted_moves = []
        board = game_state.board
        
        for move in play_moves:
            weight = 1.0
            point = move.point
            
            for neighbor in point.neighbors():
                if 1 <= neighbor.row <= board.num_rows and \
                   1 <= neighbor.col <= board.num_cols:
                    stone = board.get(neighbor)
                    if stone is not None:
                        weight += 3.0  
                        break
            
            center_row = (board.num_rows + 1) / 2.0
            center_col = (board.num_cols + 1) / 2.0
            dist = abs(point.row - center_row) + abs(point.col - center_col)
            weight += max(0, 2.0 - dist / 2.0)
            
            weighted_moves.append((move, weight))
        
        if weighted_moves:
            total_weight = sum(w for _, w in weighted_moves)
            r = random.uniform(0, total_weight)
            current = 0
            for move, weight in weighted_moves:
                current += weight
                if r <= current:
                    return move
            return weighted_moves[-1][0]
        else:
            return random.choice(candidates)

    def _evaluate_position(self, game_state, initial_player):
        """
        评估棋盘位置（用于深度限制时的启发式评估）。

        简单策略：统计棋盘上双方棋子数量。

        Args:
            game_state: 游戏状态
            initial_player: 初始玩家

        Returns:
            评估值（1=初始玩家有利，0=对手有利，0.5=平衡）
        """
        board = game_state.board
        black_stones = 0
        white_stones = 0
        
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                stone = board.get(Point(row, col))
                if stone == Player.black:
                    black_stones += 1
                elif stone == Player.white:
                    white_stones += 1
        
        if initial_player == Player.black:
            if black_stones > white_stones:
                return 0.7
            elif white_stones > black_stones:
                return 0.3
            else:
                return 0.5
        else: 
            if white_stones > black_stones:
                return 0.7
            elif black_stones > white_stones:
                return 0.3
            else:
                return 0.5
