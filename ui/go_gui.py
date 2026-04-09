"""
围棋图形界面（Tkinter）。

功能：
- 显示棋盘和棋子
- 支持鼠标点击落子
- 显示当前回合、提子数、终局结果
- 支持新游戏、悔棋、停一手、认输
- 支持人类 vs 智能体、智能体 vs 智能体（也支持人类 vs 人类）
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result

from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent


AgentFactory = Callable[[], object]


class GoGUI:
    """围棋图形界面控制器。"""

    def __init__(
        self,
        board_size: int = 9,
        black_type: str = "human",
        white_type: str = "mcts",
    ) -> None:
        self.root = tk.Tk()
        self.root.title("围棋 AI 对弈")

        # 棋盘绘制参数
        self.cell_size = 48
        self.margin = 32
        self.stone_radius = 18

        # 默认设置
        self.board_size_var = tk.IntVar(value=board_size)
        self.mode_var = tk.StringVar(value=self._infer_mode(black_type, white_type))
        self.black_player_type = tk.StringVar(value=black_type)
        self.white_player_type = tk.StringVar(value=white_type)

        self.status_var = tk.StringVar(value="准备开始新对局")
        self.turn_var = tk.StringVar(value="当前回合：黑")
        self.capture_var = tk.StringVar(value="提子：黑 0 / 白 0")

        self.game_state = GameState.new_game(self.board_size_var.get())
        self.black_captures = 0
        self.white_captures = 0
        self.move_count = 0
        self.history: list[tuple[GameState, int, int, int]] = []
        self._ai_job = None

        self.agent_factories: dict[str, AgentFactory] = {
            "random": lambda: RandomAgent(),
            "mcts": lambda: MCTSAgent(num_rounds=250),
            "minimax": lambda: MinimaxAgent(max_depth=3),
        }
        self.black_agent = None
        self.white_agent = None

        self._build_layout()
        self.start_new_game()

    def _infer_mode(self, black_type: str, white_type: str) -> str:
        """根据黑白双方类型推断默认模式。"""
        black_is_human = black_type == "human"
        white_is_human = white_type == "human"
        if black_is_human and white_is_human:
            return "human_vs_human"
        if black_is_human or white_is_human:
            return "human_vs_ai"
        return "ai_vs_ai"

    def _build_layout(self) -> None:
        """构建界面布局。"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="对局设置", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        ttk.Label(left, text="棋盘大小").grid(row=0, column=0, sticky="w")
        size_box = ttk.Combobox(
            left,
            textvariable=self.board_size_var,
            values=[5, 7, 9, 13],
            width=10,
            state="readonly",
        )
        size_box.grid(row=1, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(left, text="模式").grid(row=2, column=0, sticky="w")
        mode_box = ttk.Combobox(
            left,
            textvariable=self.mode_var,
            values=["human_vs_ai", "ai_vs_ai", "human_vs_human"],
            width=14,
            state="readonly",
        )
        mode_box.grid(row=3, column=0, sticky="ew", pady=(2, 10))
        mode_box.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(left, text="黑方").grid(row=4, column=0, sticky="w")
        self.black_box = ttk.Combobox(
            left,
            textvariable=self.black_player_type,
            values=["human", "random", "mcts", "minimax"],
            width=14,
            state="readonly",
        )
        self.black_box.grid(row=5, column=0, sticky="ew", pady=(2, 10))
        self.black_box.bind("<<ComboboxSelected>>", self._on_player_setting_change)

        ttk.Label(left, text="白方").grid(row=6, column=0, sticky="w")
        self.white_box = ttk.Combobox(
            left,
            textvariable=self.white_player_type,
            values=["human", "random", "mcts", "minimax"],
            width=14,
            state="readonly",
        )
        self.white_box.grid(row=7, column=0, sticky="ew", pady=(2, 10))
        self.white_box.bind("<<ComboboxSelected>>", self._on_player_setting_change)

        ttk.Button(left, text="新游戏", command=self.start_new_game).grid(
            row=8, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(left, text="悔棋", command=self.undo_move).grid(
            row=9, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(left, text="停一手", command=self.pass_turn).grid(
            row=10, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(left, text="认输", command=self.resign).grid(
            row=11, column=0, sticky="ew"
        )

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, width=640, height=640, bg="#D9A35B")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_board_click)

        info = ttk.Frame(right, padding=(0, 8, 0, 0))
        info.grid(row=1, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)

        ttk.Label(info, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.turn_var).grid(row=1, column=0, sticky="w")
        ttk.Label(info, textvariable=self.capture_var).grid(row=2, column=0, sticky="w")

    def _on_mode_change(self, _event=None) -> None:
        """根据模式自动设置双方角色。"""
        mode = self.mode_var.get()
        if mode == "human_vs_ai":
            self.black_player_type.set("human")
            if self.white_player_type.get() == "human":
                self.white_player_type.set("mcts")
        elif mode == "ai_vs_ai":
            if self.black_player_type.get() == "human":
                self.black_player_type.set("mcts")
            if self.white_player_type.get() == "human":
                self.white_player_type.set("minimax")
        elif mode == "human_vs_human":
            self.black_player_type.set("human")
            self.white_player_type.set("human")
        self._on_player_setting_change()

    def _on_player_setting_change(self, _event=None) -> None:
        """玩家类型发生变化后，刷新智能体配置。"""
        self._refresh_agents()
        self._maybe_schedule_ai_move()

    def start_new_game(self) -> None:
        """初始化新对局。"""
        self._cancel_ai_job()
        size = self.board_size_var.get()
        self.game_state = GameState.new_game(size)
        self.black_captures = 0
        self.white_captures = 0
        self.move_count = 0
        self.history = [
            (self.game_state, self.black_captures, self.white_captures, self.move_count)
        ]
        self._refresh_agents()
        self._sync_status("新对局开始")
        self._redraw_board()
        self._maybe_schedule_ai_move()

    def _refresh_agents(self) -> None:
        """根据设置创建智能体实例。"""
        self.black_agent = self._create_agent(self.black_player_type.get())
        self.white_agent = self._create_agent(self.white_player_type.get())

    def _create_agent(self, kind: str):
        """创建指定类型的智能体。"""
        if kind == "human":
            return None
        factory = self.agent_factories.get(kind)
        if factory is None:
            return None
        return factory()

    def undo_move(self) -> None:
        """悔棋：回退一步。"""
        if len(self.history) <= 1:
            self.status_var.set("已在开局，无法悔棋")
            return

        self._cancel_ai_job()
        self.history.pop()
        state, black_cap, white_cap, move_count = self.history[-1]
        self.game_state = state
        self.black_captures = black_cap
        self.white_captures = white_cap
        self.move_count = move_count
        self._sync_status("已悔棋")
        self._redraw_board()
        self._maybe_schedule_ai_move()

    def pass_turn(self) -> None:
        """当前玩家停一手。"""
        if self.game_state.is_over():
            return
        if not self._is_human_turn():
            self.status_var.set("当前为 AI 回合")
            return
        self._apply_move(Move.pass_turn())

    def resign(self) -> None:
        """当前玩家认输。"""
        if self.game_state.is_over():
            return
        if not self._is_human_turn():
            self.status_var.set("当前为 AI 回合")
            return
        self._apply_move(Move.resign())

    def _on_board_click(self, event) -> None:
        """处理鼠标点击落子。"""
        if self.game_state.is_over():
            return
        if not self._is_human_turn():
            self.status_var.set("当前为 AI 回合")
            return

        point = self._pixel_to_point(event.x, event.y)
        if point is None:
            return

        move = Move.play(point)
        if not self.game_state.is_valid_move(move):
            self.status_var.set(f"非法落子：({point.row}, {point.col})")
            return

        self._apply_move(move)

    def _pixel_to_point(self, x: int, y: int) -> Point | None:
        """将像素坐标映射为棋盘坐标。"""
        size = self.game_state.board.num_rows
        grid_len = self.cell_size * (size - 1)
        left = self.margin
        top = self.margin
        right = left + grid_len
        bottom = top + grid_len

        tolerance = self.cell_size * 0.4
        if x < left - tolerance or x > right + tolerance:
            return None
        if y < top - tolerance or y > bottom + tolerance:
            return None

        col = round((x - left) / self.cell_size) + 1
        row = round((y - top) / self.cell_size) + 1

        if not (1 <= row <= size and 1 <= col <= size):
            return None
        return Point(row=row, col=col)

    def _is_human_turn(self) -> bool:
        """判断当前是否轮到人类玩家。"""
        if self.game_state.next_player == Player.black:
            return self.black_player_type.get() == "human"
        return self.white_player_type.get() == "human"

    def _get_current_agent(self):
        """获取当前轮次的智能体对象。"""
        if self.game_state.next_player == Player.black:
            return self.black_agent
        return self.white_agent

    def _apply_move(self, move: Move) -> None:
        """应用一步棋并更新状态。"""
        if not self.game_state.is_valid_move(move):
            return

        player = self.game_state.next_player
        opponent = player.other
        before_count = self._count_stones(self.game_state, opponent)

        next_state = self.game_state.apply_move(move)

        captured = 0
        if move.is_play:
            after_count = self._count_stones(next_state, opponent)
            captured = max(0, before_count - after_count)

        if player == Player.black:
            self.black_captures += captured
        else:
            self.white_captures += captured

        self.game_state = next_state
        self.move_count += 1
        self.history.append(
            (self.game_state, self.black_captures, self.white_captures, self.move_count)
        )

        if self.game_state.is_over():
            self._sync_status(self._build_game_over_message())
        else:
            last_desc = self._describe_move(player, move)
            self._sync_status(last_desc)

        self._redraw_board()
        self._maybe_schedule_ai_move()

    def _count_stones(self, state: GameState, player: Player) -> int:
        """统计某一方在棋盘上的棋子数量。"""
        total = 0
        board = state.board
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) == player:
                    total += 1
        return total

    def _describe_move(self, player: Player, move: Move) -> str:
        """生成落子说明文本。"""
        player_name = "黑" if player == Player.black else "白"
        if move.is_pass:
            return f"{player_name} 方选择停一手"
        if move.is_resign:
            return f"{player_name} 方认输"
        return f"{player_name} 方落子：({move.point.row}, {move.point.col})"

    def _build_game_over_message(self) -> str:
        """生成终局信息。"""
        winner = self.game_state.winner()
        if self.game_state.last_move is not None and self.game_state.last_move.is_resign:
            winner_name = "黑" if winner == Player.black else "白"
            return f"终局：{winner_name} 方因对手认输获胜"

        result = compute_game_result(self.game_state)
        winner_name = "黑" if result.winner == Player.black else "白"
        return (
            f"终局：{winner_name} 胜，"
            f"黑 {result.b:.1f} - 白 {result.w + result.komi:.1f}"
        )

    def _sync_status(self, status: str) -> None:
        """刷新状态栏信息。"""
        self.status_var.set(status)
        if self.game_state.is_over():
            self.turn_var.set("当前回合：对局结束")
        else:
            side = "黑" if self.game_state.next_player == Player.black else "白"
            self.turn_var.set(f"当前回合：{side}")

        self.capture_var.set(
            f"提子：黑 {self.black_captures} / 白 {self.white_captures}"
        )

    def _redraw_board(self) -> None:
        """重绘棋盘与棋子。"""
        board = self.game_state.board
        size = board.num_rows
        canvas_size = self.margin * 2 + self.cell_size * (size - 1)
        self.canvas.config(width=canvas_size, height=canvas_size)
        self.canvas.delete("all")

        start = self.margin
        end = self.margin + self.cell_size * (size - 1)

        for i in range(size):
            pos = self.margin + i * self.cell_size
            self.canvas.create_line(start, pos, end, pos, width=1)
            self.canvas.create_line(pos, start, pos, end, width=1)

        for row in range(1, size + 1):
            for col in range(1, size + 1):
                point = Point(row=row, col=col)
                stone = board.get(point)
                if stone is None:
                    continue
                x, y = self._point_to_pixel(point)
                color = "black" if stone == Player.black else "white"
                outline = "#222" if stone == Player.black else "#888"
                self.canvas.create_oval(
                    x - self.stone_radius,
                    y - self.stone_radius,
                    x + self.stone_radius,
                    y + self.stone_radius,
                    fill=color,
                    outline=outline,
                    width=1,
                )

        if self.game_state.last_move is not None and self.game_state.last_move.is_play:
            lx, ly = self._point_to_pixel(self.game_state.last_move.point)
            self.canvas.create_oval(
                lx - 4,
                ly - 4,
                lx + 4,
                ly + 4,
                fill="#d23",
                outline="",
            )

    def _point_to_pixel(self, point: Point) -> tuple[int, int]:
        """将棋盘坐标映射为像素位置。"""
        x = self.margin + (point.col - 1) * self.cell_size
        y = self.margin + (point.row - 1) * self.cell_size
        return x, y

    def _maybe_schedule_ai_move(self) -> None:
        """若当前轮到 AI，则安排其自动走子。"""
        self._cancel_ai_job()
        if self.game_state.is_over():
            return

        agent = self._get_current_agent()
        if agent is None:
            return

        self._ai_job = self.root.after(120, self._run_ai_turn)

    def _run_ai_turn(self) -> None:
        """执行 AI 回合。"""
        self._ai_job = None
        if self.game_state.is_over():
            return

        agent = self._get_current_agent()
        if agent is None:
            return

        try:
            move = agent.select_move(self.game_state)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("AI 错误", f"智能体落子失败：{exc}")
            return

        if move is None or not self.game_state.is_valid_move(move):
            move = Move.pass_turn()

        self._apply_move(move)

    def _cancel_ai_job(self) -> None:
        """取消已计划的 AI 任务。"""
        if self._ai_job is not None:
            self.root.after_cancel(self._ai_job)
            self._ai_job = None

    def run(self) -> None:
        """启动主循环。"""
        self.root.mainloop()


def main() -> None:
    """程序入口。"""
    app = GoGUI()
    app.run()


def launch_gui(
    board_size: int = 9,
    black_type: str = "human",
    white_type: str = "mcts",
) -> None:
    """供外部模块调用的 GUI 启动函数。"""
    app = GoGUI(
        board_size=board_size,
        black_type=black_type,
        white_type=white_type,
    )
    app.run()


if __name__ == "__main__":
    main()
