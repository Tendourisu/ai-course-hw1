"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

import math
import random

from dlgo.goboard import GameState, Move
from dlgo.gotypes import Point
from dlgo.scoring import compute_game_result

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
            raise ValueError("best_child() called on a node without children.")

        best_score = float("-inf")
        best_children = []
        log_parent_visits = math.log(max(1, self.visit_count))

        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploration = c * math.sqrt(log_parent_visits / child.visit_count)
                score = child.value + exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        if self.is_terminal():
            return self
        if self.children:
            return random.choice(self.children)

        legal_moves = [move for move in self.game_state.legal_moves() if not move.is_resign]
        if not legal_moves:
            legal_moves = [Move.resign()]

        random.shuffle(legal_moves)
        for move in legal_moves:
            child_state = self.game_state.apply_move(move)
            child = MCTSNode(child_state, parent=self, move=move)
            self.children.append(child)

        return random.choice(self.children)

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            self.parent.backup(1.0 - value)


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
        self.rollout_depth = 30

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

        if game_state.is_over():
            return Move.pass_turn()

        legal_moves = [move for move in game_state.legal_moves() if not move.is_resign]
        if len(legal_moves) == 1:
            return legal_moves[0]

        exploration = math.sqrt(2.0) * max(0.1, self.temperature)

        for _ in range(self.num_rounds):
            node = root

            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child(c=exploration)

            if not node.is_terminal():
                node = node.expand()

            rollout_value = self._simulate(node.game_state)
            node.backup(1.0 - rollout_value)

        return self._select_best_move(root)

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
        rollout_player = game_state.next_player
        current_state = game_state

        for _ in range(self.rollout_depth):
            if current_state.is_over():
                break

            move = self._select_rollout_move(current_state)
            current_state = current_state.apply_move(move)

        winner = self._winner_for_rollout(current_state)
        if winner is None:
            return 0.5
        if winner == rollout_player:
            return 1.0
        return 0.0

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        if not root.children:
            legal_moves = [move for move in root.game_state.legal_moves() if not move.is_resign]
            if legal_moves:
                return random.choice(legal_moves)
            return Move.resign()

        best_child = max(
            root.children,
            key=lambda child: (child.visit_count, child.value),
        )
        return best_child.move

    def _select_rollout_move(self, game_state):
        """Choose a rollout move with simple local heuristics."""
        legal_moves = [move for move in game_state.legal_moves() if not move.is_resign]
        play_moves = [move for move in legal_moves if move.is_play]

        if not play_moves:
            return Move.pass_turn()

        scored_moves = []
        for move in play_moves:
            score = self._score_rollout_move(game_state, move)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        candidate_count = min(3, len(scored_moves))
        top_candidates = [move for _, move in scored_moves[:candidate_count]]

        # Keep some randomness so rollouts do not collapse into a fixed policy.
        if random.random() < 0.15 and any(move.is_pass for move in legal_moves):
            return Move.pass_turn()
        return random.choice(top_candidates)

    def _score_rollout_move(self, game_state, move):
        """Heuristic score for rollout move ordering."""
        next_state = game_state.apply_move(move)
        own_string = next_state.board.get_go_string(move.point)
        liberties = own_string.num_liberties if own_string is not None else 0

        score = liberties
        score += 4 * self._count_captured_stones(game_state, next_state)
        score += self._count_adjacent_allies(game_state, move)
        score += 0.5 * self._count_adjacent_opponents(game_state, move)

        if liberties <= 1:
            score -= 3

        center_row = (game_state.board.num_rows + 1) / 2.0
        center_col = (game_state.board.num_cols + 1) / 2.0
        distance_to_center = abs(move.point.row - center_row) + abs(move.point.col - center_col)
        score -= 0.1 * distance_to_center

        return score

    def _count_captured_stones(self, game_state, next_state):
        """Count how many opponent stones are captured by a move."""
        player = game_state.next_player
        opponent = player.other
        before = self._count_stones(game_state, opponent)
        after = self._count_stones(next_state, opponent)
        return before - after

    def _count_stones(self, game_state, player):
        """Count stones of one color on the board."""
        total = 0
        for row in range(1, game_state.board.num_rows + 1):
            for col in range(1, game_state.board.num_cols + 1):
                if game_state.board.get(Point(row, col)) == player:
                    total += 1
        return total

    def _count_adjacent_allies(self, game_state, move):
        """Count friendly neighboring stones."""
        count = 0
        for neighbor in move.point.neighbors():
            if not game_state.board.is_on_grid(neighbor):
                continue
            if game_state.board.get(neighbor) == game_state.next_player:
                count += 1
        return count

    def _count_adjacent_opponents(self, game_state, move):
        """Count opponent neighboring stones."""
        count = 0
        for neighbor in move.point.neighbors():
            if not game_state.board.is_on_grid(neighbor):
                continue
            if game_state.board.get(neighbor) == game_state.next_player.other:
                count += 1
        return count

    def _winner_for_rollout(self, game_state):
        """Get a winner at the end of a rollout or by quick evaluation."""
        if game_state.is_over():
            return game_state.winner()
        return compute_game_result(game_state).winner
