"""
围棋 AI 图形化用户界面 (GUI)

使用 Tkinter 实现 5×5 棋盘的可视化和交互
支持人类玩家与 AI 对弈
"""

import tkinter as tk
from tkinter import messagebox, font
import threading
import time
import argparse
from dlgo import GameState, Player, Point
from dlgo.goboard import Move
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent

__all__ = ["GoboardGUI"]


class GoboardGUI:
    """
    围棋棋盘 GUI
    
    功能：
    - 显示 5×5 棋盘
    - 支持鼠标点击落子
    - 显示游戏信息（当前玩家、提子数、步数等）
    - 支持与多种 AI 对弈
    - 新游戏按钮
    """

    def __init__(self, root, board_size=5, ai_agent="mcts"):
        """
        初始化 GUI

        Args:
            root: Tkinter 根窗口
            board_size: 棋盘大小（默认 5×5）
            ai_agent: AI 类型 ("random", "mcts", "minimax")
        """
        self.root = root
        self.root.title("围棋 AI - 人机对弈")
        self.board_size = board_size
        self.ai_agent_type = ai_agent
        
        # 初始化游戏状态
        self.game = GameState.new_game(board_size)
        self.move_history = []
        
        # AI 智能体
        self._init_ai_agent()
        
        # 绘图参数
        self.margin = 40
        self.cell_size = 60
        self.board_width = board_size * self.cell_size + 2 * self.margin
        self.board_height = board_size * self.cell_size + 2 * self.margin
        
        # 棋子半径
        self.stone_radius = self.cell_size // 2 - 3
        
        # 是否轮到 AI
        self.is_ai_thinking = False
        self.human_color = Player.black
        self.ai_color = Player.white
        
        # 构建 UI
        self._build_ui()
        
        # 初始显示
        self._update_display()

    def _init_ai_agent(self):
        """初始化 AI 智能体"""
        if self.ai_agent_type == "random":
            self.ai_agent = RandomAgent()
        elif self.ai_agent_type == "mcts":
            self.ai_agent = MCTSAgent(num_rounds=100)
        elif self.ai_agent_type == "minimax":
            self.ai_agent = MinimaxAgent(max_depth=2)
        else:
            self.ai_agent = RandomAgent()

    def _build_ui(self):
        """构建用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root, bg="#d4a574")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = tk.Label(
            main_frame, 
            text="围棋 AI 对弈系统",
            font=("Arial", 16, "bold"),
            bg="#d4a574"
        )
        title_label.pack(pady=10)
        
        # 左上：棋盘
        canvas_frame = tk.Frame(main_frame, bg="white", relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.board_width,
            height=self.board_height,
            bg="#d4a574",
            relief=tk.SUNKEN,
            bd=1
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # 右侧：信息面板
        info_frame = tk.Frame(main_frame, bg="#d4a574", relief=tk.SUNKEN, bd=2)
        info_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 游戏信息
        info_title = tk.Label(
            info_frame,
            text="游戏信息",
            font=("Arial", 12, "bold"),
            bg="#d4a574"
        )
        info_title.pack(pady=5)
        
        # 当前玩家
        self.player_label = tk.Label(
            info_frame,
            text="当前玩家: 黑棋 (人类)",
            font=("Arial", 10),
            bg="#d4a574"
        )
        self.player_label.pack(pady=5)
        
        # 步数
        self.move_count_label = tk.Label(
            info_frame,
            text="步数: 0",
            font=("Arial", 10),
            bg="#d4a574"
        )
        self.move_count_label.pack(pady=5)
        
        # 游戏状态
        self.status_label = tk.Label(
            info_frame,
            text="状态: 游戏进行中...",
            font=("Arial", 10),
            bg="#d4a574",
            fg="green"
        )
        self.status_label.pack(pady=5)
        
        # 分隔符
        separator = tk.Frame(info_frame, height=2, bg="black")
        separator.pack(fill=tk.X, pady=10)
        
        # 控制按钮
        button_title = tk.Label(
            info_frame,
            text="控制",
            font=("Arial", 12, "bold"),
            bg="#d4a574"
        )
        button_title.pack(pady=5)
        
        # 新游戏按钮
        new_game_btn = tk.Button(
            info_frame,
            text="新游戏 (黑棋)",
            command=self._new_game_human_black,
            width=20,
            bg="#90EE90"
        )
        new_game_btn.pack(pady=5)
        
        # 新游戏按钮 (白棋)
        new_game_btn2 = tk.Button(
            info_frame,
            text="新游戏 (白棋)",
            command=self._new_game_human_white,
            width=20,
            bg="#FFB6C1"
        )
        new_game_btn2.pack(pady=5)
        
        # Pass 按钮
        pass_btn = tk.Button(
            info_frame,
            text="停一手 (Pass)",
            command=self._human_pass,
            width=20
        )
        pass_btn.pack(pady=5)
        
        # Undo 按钮（悔棋）
        undo_btn = tk.Button(
            info_frame,
            text="悔棋 (Undo)",
            command=self._undo_move,
            width=20
        )
        undo_btn.pack(pady=5)
        
        # AI 选择标签
        ai_label = tk.Label(
            info_frame,
            text="AI 算法: " + self.ai_agent_type.upper(),
            font=("Arial", 9),
            bg="#d4a574",
            fg="blue"
        )
        ai_label.pack(pady=5)
        
        # 退出按钮
        exit_btn = tk.Button(
            info_frame,
            text="退出",
            command=self.root.quit,
            width=20,
            bg="#FFB6C1"
        )
        exit_btn.pack(pady=5)

    def _draw_board(self):
        """绘制棋盘"""
        self.canvas.delete("all")
        
        # 绘制网格线
        for i in range(self.board_size):
            x1 = self.margin + i * self.cell_size
            y1 = self.margin
            x2 = self.margin + i * self.cell_size
            y2 = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
            
            y1 = self.margin + i * self.cell_size
            x1 = self.margin
            y2 = self.margin + i * self.cell_size
            x2 = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        
        # 绘制棋子
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                point = Point(row, col)
                stone = self.game.board.get(point)
                
                x = self.margin + (col - 1) * self.cell_size
                y = self.margin + (row - 1) * self.cell_size
                
                if stone == Player.black:
                    self.canvas.create_oval(
                        x - self.stone_radius, y - self.stone_radius,
                        x + self.stone_radius, y + self.stone_radius,
                        fill="black", outline="black"
                    )
                elif stone == Player.white:
                    self.canvas.create_oval(
                        x - self.stone_radius, y - self.stone_radius,
                        x + self.stone_radius, y + self.stone_radius,
                        fill="white", outline="black", width=2
                    )

    def _update_display(self):
        """更新界面显示"""
        self._draw_board()
        
        # 更新标签
        if self.game.is_over():
            winner = self.game.winner()
            if winner is None:
                status_text = "状态: 平局"
                status_color = "orange"
            else:
                winner_name = "黑棋" if winner == Player.black else "白棋"
                status_text = f"状态: {winner_name} 胜利！"
                status_color = "green"
            self.status_label.config(text=status_text, fg=status_color)
        else:
            player_name = "黑棋 (人类)" if self.game.next_player == self.human_color \
                          else f"白棋 (AI - {self.ai_agent_type.upper()})"
            self.player_label.config(text=f"当前玩家: {player_name}")
            self.status_label.config(text="状态: 游戏进行中...", fg="green")
            
            # 如果轮到 AI，启动 AI 思考线程
            if self.game.next_player == self.ai_color and not self.is_ai_thinking:
                self.is_ai_thinking = True
                threading.Thread(target=self._ai_move, args=(self.game,), daemon=True).start()
        
        self.move_count_label.config(text=f"步数: {len(self.move_history)}")

    def _on_canvas_click(self, event):
        """处理棋盘点击事件"""
        if self.game.is_over() or self.is_ai_thinking:
            return
        
        # 检查是否轮到人类玩家
        if self.game.next_player != self.human_color:
            return
        
        # 将像素坐标转换为棋盘坐标
        col = int((event.x - self.margin + self.cell_size / 2) // self.cell_size) + 1
        row = int((event.y - self.margin + self.cell_size / 2) // self.cell_size) + 1
        
        # 边界检查
        if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
            return
        
        # 尝试落子
        move = Move.play(Point(row, col))
        if self.game.is_valid_move(move):
            self.game = self.game.apply_move(move)
            self.move_history.append(move)
            self._update_display()
        else:
            messagebox.showwarning("非法移动", "这个位置不能落子！")

    def _ai_move(self, game_state_when_started):
        """AI 落子（在独立线程中运行）"""
        try:
            start_time = time.time()
            # 注意：传入的是启动线程时的那份快照状态卷
            move = self.ai_agent.select_move(game_state_when_started)
            
            # 使用内嵌函数安全地在主线程更新状态
            def apply_and_update():
                try:
                    # 并发防御：检查游戏是否在 AI 思考期间被用户重置
                    if self.game is not game_state_when_started:
                        return
                    
                    # 应用 AI 的棋步
                    self.game = self.game.apply_move(move)
                    self.move_history.append(move)
                    
                    # 给出会意提示
                    if move.is_pass:
                        messagebox.showinfo("AI 动作", f"AI ({self.ai_agent_type.upper()}) 选择了 停一手(Pass)")
                    elif move.is_resign:
                        messagebox.showinfo("AI 动作", f"AI ({self.ai_agent_type.upper()}) 选择了 认输")
                        
                    self._update_display()
                finally:
                    self.is_ai_thinking = False

            # 派发到主线程更新 GUI
            self.root.after(0, apply_and_update)
            
        except Exception as e:
            def show_error():
                messagebox.showerror("AI 错误", f"AI 出错: {str(e)}")
                self.is_ai_thinking = False
            self.root.after(0, show_error)

    def _human_pass(self):
        """人类玩家停一手"""
        if self.game.is_over():
            messagebox.showinfo("游戏已结束", "游戏已结束，请开始新游戏")
            return
        
        if self.game.next_player != self.human_color:
            messagebox.showinfo("不是你的回合", "现在轮到 AI")
            return
        
        move = Move.pass_turn()
        self.game = self.game.apply_move(move)
        self.move_history.append(move)
        self._update_display()

    def _undo_move(self):
        """悔棋（撤销）真正实现"""
        if self.is_ai_thinking:
            messagebox.showwarning("请稍候", "AI 正在思考，无法强行打断从而悔棋！")
            return
            
        if self.game.previous_state is None:
            messagebox.showinfo("无法悔棋", "当前已经是初始状态了。")
            return
            
        # 逻辑：如果是人类的局，应该同时连退两步：撤销 AI 的走子 和 撤销自己的走子
        if self.game.next_player == self.human_color and self.game.previous_state.previous_state is not None:
            self.game = self.game.previous_state.previous_state
            self.move_history.pop()  # 弹出两次
            self.move_history.pop()
        else:
            # 否则退一步即可
            self.game = self.game.previous_state
            self.move_history.pop()
            
        self._update_display()

    def _new_game_human_black(self):
        """开始新游戏（人类执黑）"""
        self.game = GameState.new_game(self.board_size)
        self.move_history = []
        self.human_color = Player.black
        self.ai_color = Player.white
        self._update_display()
        messagebox.showinfo("新游戏开始", "新游戏已开始，黑棋（你）先落子")

    def _new_game_human_white(self):
        """开始新游戏（人类执白）"""
        self.game = GameState.new_game(self.board_size)
        self.move_history = []
        self.human_color = Player.white
        self.ai_color = Player.black
        self.is_ai_thinking = False  # _update_display 内部会自动触发 AI
        self._update_display()
        messagebox.showinfo("新游戏开始", "新游戏已开始，AI（黑棋）由于第一步耗时可能发呆，请稍候。")


def main():
    """主函数：启动 GUI"""
    parser = argparse.ArgumentParser(description="围棋AI 人机对弈GUI")
    parser.add_argument('--agent', '-a', type=str, default='mcts', 
                        choices=['random', 'mcts', 'minimax'],
                        help="选择对弈的AI类型：random, mcts 或 minimax (默认: mcts)")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1000x650")
    root.resizable(False, False)
    
    # 从命令行参数读取选择的 AI 类型
    gui = GoboardGUI(root, board_size=5, ai_agent=args.agent)
    
    root.mainloop()


if __name__ == "__main__":
    main()
