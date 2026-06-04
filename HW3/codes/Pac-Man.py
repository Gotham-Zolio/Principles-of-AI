import os
import argparse
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod

matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════════
# 地图 & 游戏常量
# ═══════════════════════════════════════════════════════════════
UNIT  = 100
Map_H = 5
Map_W = 5

BEAN_POSITIONS  = [(0, 2), (2, 3)]           # 豆子 (row, col)
GHOST_LIST      = [(1, 2), (2, 1), (3, 3)]   # 幽灵（有序，GUI 渲染用）
GHOST_SET       = set(GHOST_LIST)             # O(1) 碰撞检测
START_POS       = (0, 0)
GOAL_POS        = (4, 4)

STEP_COST       = -1
WALL_PENALTY    = -10
GHOST_PENALTY   = -100
BEAN_REWARD     = +10
GOAL_REWARD     = +100
NO_BEAN_PENALTY = -50
GAMMA           = 0.99

N_BEANS   = len(BEAN_POSITIONS)    # 2
N_MASKS   = 1 << N_BEANS           # 4
N_STATES  = Map_H * Map_W * N_MASKS  # 100
N_ACTIONS = 4                        # 0=上 1=下 2=左 3=右

DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]
ACTION_ARROW = ['↑', '↓', '←', '→']

map_state = np.array([
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
])

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 状态编解码 & 确定性转移函数
# ═══════════════════════════════════════════════════════════════

def encode(r: int, c: int, b: int) -> int:
    return (r * Map_W + c) * N_MASKS + b


def decode(sid: int):
    b  = sid % N_MASKS
    rc = sid // N_MASKS
    return rc // Map_W, rc % Map_W, b


def is_terminal(r: int, c: int) -> bool:
    return (r, c) in GHOST_SET or (r, c) == GOAL_POS


def step_logic(r: int, c: int, beans: int, action: int):
    nr, nc = r + DR[action], c + DC[action]

    if not (0 <= nr < Map_H and 0 <= nc < Map_W):
        return r, c, beans, WALL_PENALTY, False

    nb     = beans
    reward = STEP_COST

    for i, (br, bc) in enumerate(BEAN_POSITIONS):
        if nr == br and nc == bc and not (nb >> i & 1):
            nb     |= 1 << i
            reward += BEAN_REWARD

    if (nr, nc) in GHOST_SET:
        return nr, nc, nb, reward + GHOST_PENALTY, True

    if (nr, nc) == GOAL_POS:
        bonus = GOAL_REWARD if nb == N_MASKS - 1 else NO_BEAN_PENALTY
        return nr, nc, nb, reward + bonus, True

    return nr, nc, nb, reward, False


def grid_to_canvas(col, row):
    origin = np.array([UNIT / 2, UNIT / 2])
    return origin[0] + col * UNIT, origin[1] + row * UNIT


# ═══════════════════════════════════════════════════════════════
# 训练环境
# ═══════════════════════════════════════════════════════════════

class PacManEnv:

    def reset(self) -> int:
        self._r, self._c, self._b = START_POS[0], START_POS[1], 0
        return encode(self._r, self._c, self._b)

    def step(self, action: int):

        nr, nc, nb, reward, done = step_logic(
            self._r, self._c, self._b, action)
        if done:
            self._r, self._c, self._b = START_POS[0], START_POS[1], 0
        else:
            self._r, self._c, self._b = nr, nc, nb
        return encode(nr, nc, nb), reward, done


# ═══════════════════════════════════════════════════════════════
# 有模型 RL —— 值迭代（动态规划）
# ═══════════════════════════════════════════════════════════════

def value_iteration(gamma: float = GAMMA,
                    theta: float = 1e-6,
                    max_iter: int = 10_000):

    V      = np.zeros(N_STATES)
    policy = np.zeros(N_STATES, dtype=int)

    for itr in range(max_iter):
        delta = 0.0
        for s in range(N_STATES):
            r, c, b = decode(s)
            if is_terminal(r, c):
                continue          # 吸收态，V 固定为 0

            qs = np.empty(N_ACTIONS)
            for a in range(N_ACTIONS):
                nr2, nc2, nb2, reward, done = step_logic(r, c, b, a)
                ns = encode(nr2, nc2, nb2)
                qs[a] = reward + gamma * (0.0 if done else V[ns])

            best      = float(qs.max())
            delta     = max(delta, abs(best - V[s]))
            V[s]      = best
            policy[s] = int(qs.argmax())

        if delta < theta:
            print(f"[值迭代] 第 {itr + 1} 轮收敛，Δ = {delta:.2e}")
            break

    return V, policy


# ═══════════════════════════════════════════════════════════════
# 无模型 RL —— 蒙特卡洛控制
# ═══════════════════════════════════════════════════════════════

def monte_carlo_control(n_episodes: int = 3000,
                         gamma: float = GAMMA,
                         epsilon_start: float = 1.0,
                         epsilon_min: float = 0.02,
                         max_steps: int = 500,
                         snapshot_every: int = 100):

    env  = PacManEnv()
    Q    = np.zeros((N_STATES, N_ACTIONS))
    Cnt  = np.zeros((N_STATES, N_ACTIONS), dtype=np.int32)

    episode_returns = []
    episode_lengths = []
    snapshots       = []

    for ep in range(n_episodes):
        epsilon = max(epsilon_min,
                      epsilon_start * np.exp(-5.0 * ep / n_episodes))
        state = env.reset()
        traj  = []          # (state, action, reward)

        for _ in range(max_steps):
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(Q[state].argmax())
            next_s, reward, done = env.step(action)
            traj.append((state, action, reward))
            state = next_s
            if done:
                break

        # First-Visit MC 更新（逆序累积回报）
        G       = 0.0
        visited = set()
        for s, a, rw in reversed(traj):
            G = rw + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                Cnt[s, a] += 1
                Q[s, a]   += (G - Q[s, a]) / Cnt[s, a]

        episode_returns.append(sum(rw for _, _, rw in traj))
        episode_lengths.append(len(traj))

        if ep % snapshot_every == 0 or ep == n_episodes - 1:
            snapshots.append((ep, Q.argmax(axis=1).copy()))

        if (ep + 1) % 500 == 0:
            recent = episode_returns[-500:]
            print(f"  [MC] 第 {ep+1:5d}/{n_episodes} 回合，"
                  f"近 500 回合均值 = {np.mean(recent):.1f}")

    policy = Q.argmax(axis=1)
    return Q, policy, episode_returns, episode_lengths, snapshots


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

def print_policy(policy, label: str = "最优策略"):
    """在终端打印各 beans_mask 下的策略网格"""
    print(f"\n{'═' * 52}")
    print(f"  {label}")
    print(f"{'═' * 52}")
    for b in range(N_MASKS):
        eaten = [f"豆{i+1}" for i in range(N_BEANS) if (b >> i & 1)]
        print(f"\n  [beans_mask={b:02b}  已吃: {', '.join(eaten) or '无'}]")
        for r in range(Map_H):
            line = ""
            for c in range(Map_W):
                s = encode(r, c, b)
                if   (r, c) == GOAL_POS:              line += " G "
                elif (r, c) in GHOST_SET:             line += " X "
                elif (r, c) == START_POS and b == 0:  line += " S "
                else:                                  line += f" {ACTION_ARROW[policy[s]]} "
            print(line)


def _draw_policy_ax(ax, policy, b_idx, V=None, title=""):
    """在单个 Axes 上绘制策略箭头图（可叠加 V 值热图）"""
    ax.clear()
    ax.set_xlim(0, Map_W);  ax.set_ylim(0, Map_H)
    ax.set_aspect('equal');  ax.invert_yaxis()
    ax.set_xticks(range(Map_W + 1));  ax.set_yticks(range(Map_H + 1))
    ax.grid(True, color='gray', linewidth=0.4)
    ax.set_title(title or f"beans={b_idx:02b}", fontsize=9)

    # 箭头偏移（invert_yaxis 后 y 增大方向朝下，上/下动作 ady 符号相应取反）
    adx = {0: 0,     1: 0,     2: -0.3, 3:  0.3}
    ady = {0: -0.3,  1:  0.3,  2:  0,   3:  0}

    for r in range(Map_H):
        for c in range(Map_W):
            cx, cy = c + 0.5, r + 0.5
            s = encode(r, c, b_idx)

            # V 值热图背景
            if V is not None and not is_terminal(r, c):
                v_min, v_max = V.min(), V.max()
                norm = (V[s] - v_min) / max(v_max - v_min, 1e-9)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1,
                    color=plt.cm.RdYlGn(norm), alpha=0.45, zorder=0))

            if (r, c) == GOAL_POS:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                    color='gold', alpha=0.85, zorder=1))
                ax.text(cx, cy, 'G', ha='center', va='center',
                        fontsize=11, fontweight='bold', zorder=2)
                continue

            if (r, c) in GHOST_SET:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                    color='#ffaaaa', alpha=0.85, zorder=1))
                ax.text(cx, cy, '×', ha='center', va='center',
                        color='red', fontsize=14, zorder=2)
                continue

            if (r, c) == START_POS and b_idx == 0:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                    color='lightblue', alpha=0.55, zorder=1))

            # 未吃豆子标记
            for i, (br, bc) in enumerate(BEAN_POSITIONS):
                if r == br and c == bc and not (b_idx >> i & 1):
                    ax.plot(cx, cy, 'o', color='#ffe066',
                            markersize=9, markeredgecolor='gray', zorder=3)

            # 策略箭头
            dx, dy = adx[policy[s]], ady[policy[s]]
            ax.annotate('',
                xy=(cx + dx, cy + dy),
                xytext=(cx - dx * 0.3, cy - dy * 0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                zorder=4)


def visualize_policy(policy, V=None,
                     title: str = "最优策略（箭头图）",
                     save_path: str = None):
    """4 张子图分别展示 4 种 beans_mask 下的策略"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=13)
    for b in range(N_MASKS):
        eaten = [f"豆{i+1}" for i in range(N_BEANS) if (b >> i & 1)]
        subtitle = f"已吃: {', '.join(eaten) or '无'} (mask={b:02b})"
        _draw_policy_ax(axes[b // 2, b % 2], policy, b, V, subtitle)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[策略图] 已保存: {save_path}")
    return fig


def plot_learning_curves(episode_returns, episode_lengths,
                          window: int = 50, save_path: str = None):
    """绘制学习曲线：每回合总回报 & 步数随回合数变化"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    episodes = np.arange(1, len(episode_returns) + 1)

    def smooth(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    ax1.plot(episodes, episode_returns, alpha=0.25, color='steelblue',
             label='每回合回报')
    if len(episode_returns) >= window:
        sm = smooth(episode_returns, window)
        ax1.plot(episodes[:len(sm)], sm, color='navy',
                 label=f'{window} 回合滑动均值')
    ax1.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax1.set_xlabel('回合数');  ax1.set_ylabel('总回报')
    ax1.set_title('学习曲线：每回合总回报')
    ax1.legend();  ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, episode_lengths, alpha=0.25, color='darkorange',
             label='每回合步数')
    if len(episode_lengths) >= window:
        sm2 = smooth(episode_lengths, window)
        ax2.plot(episodes[:len(sm2)], sm2, color='darkred',
                 label=f'{window} 回合滑动均值')
    ax2.set_xlabel('回合数');  ax2.set_ylabel('步数')
    ax2.set_title('学习曲线：每回合路径长度（步数）')
    ax2.legend();  ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[学习曲线] 已保存: {save_path}")
    return fig


def create_policy_animation(snapshots, save_path: str = None):
    """将 MC 训练快照制作为策略演化动画（MP4 / GIF）"""
    if not snapshots:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def animate(frame_idx):
        ep_idx, pol = snapshots[frame_idx]
        for b in range(N_MASKS):
            eaten = [f"豆{i+1}" for i in range(N_BEANS) if (b >> i & 1)]
            subtitle = f"已吃: {', '.join(eaten) or '无'} (mask={b:02b})"
            _draw_policy_ax(axes[b // 2, b % 2], pol, b, None, subtitle)
        fig.suptitle(
            f"策略演化（MC 学习过程）— 第 {ep_idx} 回合", fontsize=12)
        return []

    animate(0)
    ani = anim_mod.FuncAnimation(
        fig, animate, frames=len(snapshots),
        interval=600, blit=False, repeat=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        try:
            writer = (anim_mod.FFMpegWriter(fps=2, bitrate=1800)
                      if ext == '.mp4' else anim_mod.PillowWriter(fps=2))
            ani.save(save_path, writer=writer)
            print(f"[动画] 已保存: {save_path}")
        except Exception as e:
            print(f"[动画] {ext} 保存失败: {e}")
            if ext == '.mp4':
                gif_path = os.path.splitext(save_path)[0] + '.gif'
                try:
                    ani.save(gif_path, writer=anim_mod.PillowWriter(fps=2))
                    print(f"[动画] 已改存 GIF: {gif_path}")
                except Exception as e2:
                    print(f"[动画] GIF 保存也失败: {e2}")
    return ani


def simulate_policy(policy, label: str = "策略"):
    """按策略执行一回合，打印完整轨迹"""
    env   = PacManEnv()
    state = env.reset()
    total = 0
    steps = []

    for _ in range(200):
        r, c, b = decode(state)
        a = int(policy[state])
        steps.append((r, c, b, ACTION_ARROW[a]))
        ns, reward, done = env.step(a)
        total += reward
        state  = ns
        if done:
            r2, c2, b2 = decode(state)
            steps.append((r2, c2, b2, '■'))
            break

    print(f"\n[{label}] 轨迹（共 {len(steps)-1} 步，总回报 = {total}）:")
    for i, (r, c, b, a) in enumerate(steps):
        print(f"  {i:3d}: ({r},{c}) beans={b:02b}  {a}")

    r2, c2, b2 = steps[-1][:3]
    if (r2, c2) == GOAL_POS and b2 == N_MASKS - 1:
        print("  ✓ 成功：全部豆子已吃，到达终点！")
    elif (r2, c2) == GOAL_POS:
        print(f"  ✗ 到达终点但豆子未吃完（mask={b2:02b}）")
    elif (r2, c2) in GHOST_SET:
        print("  ✗ 碰到幽灵，失败")
    else:
        print("  ? 超出步数限制，未到终点")

    return steps, total


# ═══════════════════════════════════════════════════════════════
# tkinter 可视化（Map 类，含 TODO 1-3 实现）
# ═══════════════════════════════════════════════════════════════

class Map(tk.Tk, object):
    # ── 速度档位：(每步延迟ms, 回合间隔ms) ───────────────────────
    _SPEEDS = {
        '慢速':  (200, 600),
        '正常':  ( 80, 250),
        '快速':  ( 20, 80),
        '极速':  (  5, 20),
    }

    def __init__(self):
        super(Map, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions    = len(self.action_space)
        self.title('Pac-Man')
        # 额外 90px 高度留给状态栏
        self.geometry('{0}x{1}+400+50'.format(
            Map_W * UNIT, Map_H * UNIT + 90))

        self._init_bean_positions = list(BEAN_POSITIONS)
        self.bean_positions       = list(BEAN_POSITIONS)
        self.ghosts = [
            {'row': r, 'col': c, 'type': 'static'}
            for r, c in GHOST_LIST
        ]

        # 学习速度（默认"正常"）
        self._step_delay_ms    = 80
        self._episode_pause_ms = 250

        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=Map_H * UNIT, width=Map_W * UNIT)
        for x in range(0, Map_W * UNIT + 1, UNIT):
            self.canvas.create_line(x, 0, x, Map_H * UNIT, fill='gray')
        for y in range(0, Map_H * UNIT + 1, UNIT):
            self.canvas.create_line(0, y, Map_W * UNIT, y, fill='gray')

        def _load(name):
            path = os.path.join(_SCRIPT_DIR, name)
            return ImageTk.PhotoImage(
                Image.open(path).resize((80, 80), Image.Resampling.LANCZOS))

        self.bm_beans  = _load('beans.png')
        self.bm_ghost  = _load('ghost.png')
        self.bm_person = _load('pac-man.png')
        self.bm_flag   = _load('destination.png')

        # 终点
        gx, gy = grid_to_canvas(GOAL_POS[1], GOAL_POS[0])
        self.flag = self.canvas.create_image(
            gx, gy, image=self.bm_flag, tag='destination')

        # 豆子
        self.bean_items = []
        for i, (br, bc) in enumerate(self._init_bean_positions):
            cx, cy = grid_to_canvas(bc, br)
            item = self.canvas.create_image(
                cx, cy, image=self.bm_beans, tag=f'bean{i}')
            self.bean_items.append(item)

        # 幽灵
        self.ghost_items = []
        for i, g in enumerate(self.ghosts):
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(
                cx, cy, image=self.bm_ghost, tag=f'ghost{i}')
            self.ghost_items.append(item)

        # 吃豆人（起点）
        px, py = grid_to_canvas(START_POS[1], START_POS[0])
        self.person = self.canvas.create_image(
            px, py, image=self.bm_person)
        self.canvas.pack()

        # ── 状态栏（Canvas 下方）──────────────────────────────────
        status_frame = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X, padx=4, pady=(2, 0))

        self._info_var = tk.StringVar(value="准备就绪")
        info_lbl = tk.Label(status_frame, textvariable=self._info_var,
                            font=('Microsoft YaHei', 9), anchor='w',
                            fg='#333333')
        info_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # 速度控制按钮
        speed_frame = tk.Frame(self)
        speed_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Label(speed_frame, text='速度:', font=('Microsoft YaHei', 9)
                 ).pack(side=tk.LEFT)
        for name, (sd, ep) in self._SPEEDS.items():
            btn = tk.Button(
                speed_frame, text=name,
                font=('Microsoft YaHei', 8), width=5,
                command=lambda s=sd, e=ep: self._set_speed(s, e))
            btn.pack(side=tk.LEFT, padx=2)

    def _reset_canvas(self):
        """重建所有 canvas 元素（不含任何 sleep）"""
        self.canvas.delete(self.person)

        # 重置豆子
        self.bean_positions = list(self._init_bean_positions)
        for item in self.bean_items:
            self.canvas.delete(item)
        self.bean_items = []
        for i, (br, bc) in enumerate(self._init_bean_positions):
            cx, cy = grid_to_canvas(bc, br)
            item = self.canvas.create_image(
                cx, cy, image=self.bm_beans, tag=f'bean{i}')
            self.bean_items.append(item)

        # 重置幽灵
        self.ghosts = [{'row': r, 'col': c, 'type': 'static'}
                       for r, c in GHOST_LIST]
        for i, g in enumerate(self.ghosts):
            self.canvas.delete(self.ghost_items[i])
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(
                cx, cy, image=self.bm_ghost, tag=f'ghost{i}')
            self.ghost_items[i] = item

        # 重置吃豆人
        px, py = grid_to_canvas(START_POS[1], START_POS[0])
        self.person = self.canvas.create_image(
            px, py, image=self.bm_person)

    def reset(self):
        """带短暂 sleep 的重置（用于策略演示模式，视觉上更流畅）"""
        self.update()
        time.sleep(0.05)
        self._reset_canvas()
        self.render()
        return self.get_state()

    def reset_quick(self):
        """无阻塞重置（用于 learn 模式，不阻塞 GUI 事件循环）"""
        self._reset_canvas()
        self.render_quick()
        return self.get_state()

    def get_state(self):
        """返回简单格 ID（0-24），兼容原始接口"""
        coords = self.canvas.coords(self.person)
        col = int(coords[0] / UNIT)
        row = int(coords[1] / UNIT)
        return map_state[row, col]

    def get_full_state(self) -> int:
        """返回完整状态 ID（0-99，含 beans_mask），供 RL 策略使用"""
        coords = self.canvas.coords(self.person)
        col   = min(int(coords[0] / UNIT), Map_W - 1)
        row   = min(int(coords[1] / UNIT), Map_H - 1)
        beans = 0
        for i, (br, bc) in enumerate(self._init_bean_positions):
            if (br, bc) not in self.bean_positions:   # 该豆已被吃
                beans |= 1 << i
        return encode(row, col, beans)

    def _get_pacman_grid_pos(self):
        coords = self.canvas.coords(self.person)
        col = min(int(coords[0] / UNIT), Map_W - 1)
        row = min(int(coords[1] / UNIT), Map_H - 1)
        return row, col

    def _check_ghost_collision(self):
        row, col = self._get_pacman_grid_pos()
        return any(row == g['row'] and col == g['col'] for g in self.ghosts)

    def step(self, action):
        """执行一个动作
        action: 0=上, 1=下, 2=左, 3=右
        返回: (state, cost, done)
        """
        s    = self.canvas.coords(self.person)
        move = np.array([0, 0])
        cost = STEP_COST

        if action == 0:      # 上
            if s[1] >= UNIT:                move[1] -= UNIT
            else:                           cost = WALL_PENALTY
        elif action == 1:    # 下
            if s[1] < (Map_H - 1) * UNIT:  move[1] += UNIT
            else:                           cost = WALL_PENALTY
        elif action == 2:    # 左
            if s[0] >= UNIT:                move[0] -= UNIT
            else:                           cost = WALL_PENALTY
        elif action == 3:    # 右
            if s[0] < (Map_W - 1) * UNIT:  move[0] += UNIT
            else:                           cost = WALL_PENALTY

        self.canvas.move(self.person, move[0], move[1])
        row, col = self._get_pacman_grid_pos()

        # TODO 1: 吃到豆子，给予相应奖励，并从画布上移除该豆子
        eaten_idx = []
        for i, (br, bc) in enumerate(self.bean_positions):
            if row == br and col == bc:
                cost += BEAN_REWARD
                self.canvas.delete(self.bean_items[i])
                eaten_idx.append(i)
        for i in reversed(eaten_idx):      # 倒序删除，保持列表索引同步
            self.bean_positions.pop(i)
            self.bean_items.pop(i)

        if self._check_ghost_collision():
            # TODO 2: 碰撞幽灵，给予惩罚，并结束回合
            cost += GHOST_PENALTY
            return self.get_state(), cost, True

        # TODO 3: 到达终点，结束回合（豆子吃完给奖励，否则给惩罚）
        if row == GOAL_POS[0] and col == GOAL_POS[1]:
            cost += GOAL_REWARD if not self.bean_positions else NO_BEAN_PENALTY
            return self.get_state(), cost, True

        return self.get_state(), cost, False

    def render(self):
        time.sleep(0.1)
        self.update()
        time.sleep(0.1)

    def render_quick(self):
        """无阻塞刷新（用于实时学习中的高频刷新）"""
        self.update()

    def _set_speed(self, step_delay_ms: int, episode_pause_ms: int):
        """由速度按钮回调，修改当前速度档"""
        self._step_delay_ms    = step_delay_ms
        self._episode_pause_ms = episode_pause_ms

    # ── 实时 MC 学习 ───────────────────────────────────────────────

    def run_mc_live(self, n_episodes: int = 1000,
                    gamma: float = GAMMA,
                    epsilon_start: float = 1.0,
                    epsilon_min:   float = 0.01,
                    max_steps:     int   = 200):
        """
        在 GUI 中实时演示 MC 从零学习的全过程。
        使用 after() 调度，不阻塞主线程，可随时用速度按钮调速。
        epsilon 采用线性衰减，与无 GUI 版 monte_carlo_control 保持一致。
        """
        self._Q    = np.zeros((N_STATES, N_ACTIONS))
        self._Cnt  = np.zeros((N_STATES, N_ACTIONS), dtype=np.int32)

        self._lrn_n        = n_episodes
        self._lrn_gamma    = gamma
        self._lrn_eps0     = epsilon_start
        self._lrn_epsmin   = epsilon_min
        self._lrn_maxstep  = max_steps

        self._lrn_ep      = 0
        self._lrn_traj    = []          # 当前回合轨迹 [(s, a, r), ...]
        self._lrn_returns = []          # 每回合总回报历史
        self._lrn_cur_ret = 0.0         # 本回合累计回报（显示用）

        self.reset_quick()
        self._lrn_state = self.get_full_state()
        self._info_var.set(f"开始学习（共 {n_episodes} 回合）…")
        self.after(500, self._lrn_step)

    def _lrn_step(self):
        """每次由 after() 调用，执行学习中的一步"""
        ep      = self._lrn_ep
        # 线性衰减（与 monte_carlo_control 保持一致）：ε 从 eps0 均匀降至 epsmin
        epsilon = max(self._lrn_epsmin,
                      self._lrn_eps0 - (self._lrn_eps0 - self._lrn_epsmin)
                      * ep / max(self._lrn_n - 1, 1))
        s = self._lrn_state

        # ε-greedy 选动作
        if np.random.random() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = int(self._Q[s].argmax())

        # 执行动作（GUI 同步更新）
        _, reward, done = self.step(action)
        ns = self.get_full_state()
        self._lrn_traj.append((s, action, reward))
        self._lrn_cur_ret += reward
        self.render_quick()

        # 更新状态栏（每步刷新奖励显示）
        avg50 = (float(np.mean(self._lrn_returns[-50:]))
                 if self._lrn_returns else float('nan'))
        avg50_str = f"{avg50:.1f}" if not np.isnan(avg50) else "—"
        self._info_var.set(
            f"回合 {ep+1}/{self._lrn_n}  |  ε={epsilon:.3f}  |  "
            f"本回合: {self._lrn_cur_ret:+.0f}  |  近50均值: {avg50_str}")

        # 本回合是否结束
        episode_over = done or (len(self._lrn_traj) >= self._lrn_maxstep)
        if episode_over:
            self.after(self._episode_pause_ms, self._lrn_end_episode)
        else:
            self._lrn_state = ns
            self.after(self._step_delay_ms, self._lrn_step)

    def _lrn_end_episode(self):
        """回合结束：执行 First-Visit MC 更新，然后开始下一回合"""
        # First-Visit MC 更新
        G, visited = 0.0, set()
        for st, ac, rw in reversed(self._lrn_traj):
            G = rw + self._lrn_gamma * G
            if (st, ac) not in visited:
                visited.add((st, ac))
                self._Cnt[st, ac] += 1
                self._Q[st, ac]   += (G - self._Q[st, ac]) / self._Cnt[st, ac]

        self._lrn_returns.append(self._lrn_cur_ret)
        self._lrn_ep += 1

        if self._lrn_ep < self._lrn_n:
            self.after(10, self._lrn_start_episode)
        else:
            # 学习完成
            avg = float(np.mean(self._lrn_returns[-50:]))
            self._info_var.set(
                f"✓ 学习完成（{self._lrn_n} 回合）  |  "
                f"后50均值: {avg:.1f}  |  切换为最优策略演示…")
            policy = self._Q.argmax(axis=1)
            self.after(1200, lambda: self.run_policy(policy, delay_ms=350,
                                                     n_episodes=5))

    def _lrn_start_episode(self):
        """重置环境，开始新一回合的学习（无阻塞）"""
        self.reset_quick()
        self._lrn_traj    = []
        self._lrn_cur_ret = 0.0
        self._lrn_state   = self.get_full_state()
        self.after(self._step_delay_ms, self._lrn_step)

    def run_policy(self, policy, delay_ms: int = 380, n_episodes: int = 3):
        """在 GUI 上按给定策略执行若干回合（tkinter after 调度）"""
        self._pol      = policy
        self._pol_ep   = 0
        self._pol_n    = n_episodes
        self._pol_dly  = delay_ms
        self.reset()
        self.after(delay_ms, self._pol_step)

    def _pol_step(self):
        state  = self.get_full_state()
        action = int(self._pol[state])
        _, _, done = self.step(action)
        self.render()
        if done:
            self._pol_ep += 1
            if self._pol_ep < self._pol_n:
                self.after(800,  self._start_pol_ep)
        else:
            self.after(self._pol_dly, self._pol_step)

    def _start_pol_ep(self):
        self.reset()
        self.after(self._pol_dly, self._pol_step)


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pac-Man 强化学习：DP（值迭代）& MC（蒙特卡洛）')
    parser.add_argument('--mode', choices=['dp', 'mc', 'both', 'gui', 'learn'],
                        default='both',
                        help='运行模式：dp/mc/both/gui/learn（默认 both）'
                             '  learn=GUI实时MC学习演示')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='MC 训练回合数（默认 3000）')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'折扣因子（默认 {GAMMA}）')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='结果文件保存目录（默认脚本同目录）')
    parser.add_argument('--live-episodes', type=int, default=1000,
                        help='learn 模式的 MC 学习回合数（默认 1000，建议切换极速运行）')
    args = parser.parse_args()

    out_dir = args.save_dir or _SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    dp_policy = None
    mc_policy = None
    _ani_ref  = None   # 持有动画对象引用，防止 GC

    # ── 值迭代 ─────────────────────────────────────────────────
    if args.mode in ('dp', 'both'):
        print("\n" + "═" * 52)
        print("  有模型 RL —— 值迭代（动态规划）")
        print("═" * 52)
        V_dp, dp_policy = value_iteration(gamma=args.gamma)
        print_policy(dp_policy, "值迭代最优策略")
        simulate_policy(dp_policy, "值迭代策略")
        visualize_policy(
            dp_policy, V_dp,
            title="值迭代最优策略（含 V 值热图）",
            save_path=os.path.join(out_dir, 'dp_policy.png'))

    # ── 蒙特卡洛 ───────────────────────────────────────────────
    if args.mode in ('mc', 'both'):
        print("\n" + "═" * 52)
        print(f"  无模型 RL —— 蒙特卡洛（{args.episodes} 回合）")
        print("═" * 52)
        Q_mc, mc_policy, mc_returns, mc_lengths, mc_snaps = \
            monte_carlo_control(n_episodes=args.episodes,
                                gamma=args.gamma)
        print_policy(mc_policy, "MC 最优策略")
        simulate_policy(mc_policy, "MC 策略")
        visualize_policy(
            mc_policy,
            title="MC 最优策略（箭头图）",
            save_path=os.path.join(out_dir, 'mc_policy.png'))
        plot_learning_curves(
            mc_returns, mc_lengths,
            save_path=os.path.join(out_dir, 'learning_curves.png'))
        _ani_ref = create_policy_animation(
            mc_snaps,
            save_path=os.path.join(out_dir, 'Pac-Man-learning.mp4'))

    # ── 展示 matplotlib 图表（learn/gui 模式跳过，其余模式弹窗）──
    if args.mode not in ('gui', 'learn'):
        print("\n正在显示图表（关闭所有图表窗口后将启动 GUI 演示）...")
        try:
            plt.show()
        except Exception:
            pass

    # ── tkinter GUI 演示 ────────────────────────────────────────
    if args.mode == 'learn':
        # learn 模式：GUI 中实时演示 MC 从零学习
        env_gui = Map()
        env_gui.title('Pac-Man —— MC 实时学习演示')
        env_gui.run_mc_live(n_episodes=args.live_episodes,
                            gamma=args.gamma)
        env_gui.mainloop()
    else:   # dp / mc / both / gui 模式均打开 GUI
        gui_policy = dp_policy if dp_policy is not None else mc_policy
        env_gui = Map()
        if gui_policy is not None:
            env_gui.run_policy(gui_policy, delay_ms=350, n_episodes=3)
        env_gui.mainloop()


if __name__ == '__main__':
    main()
