"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

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
        self.result_cache = GameResultCache()
        self._root_player = None

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        if game_state.is_over():
            return Move.pass_turn()

        self._root_player = game_state.next_player
        self.result_cache = GameResultCache()

        candidate_moves = [
            move for move in self._get_ordered_moves(game_state)
            if not move.is_resign
        ]
        if not candidate_moves:
            return Move.pass_turn()
        if len(candidate_moves) == 1:
            return candidate_moves[0]

        best_score = float("-inf")
        best_move = candidate_moves[0]
        alpha = float("-inf")
        beta = float("inf")
        next_depth = max(0, self.max_depth - 1)

        for move in candidate_moves:
            next_state = game_state.apply_move(move)
            score = self.alphabeta(
                next_state, next_depth, alpha, beta, False
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

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
        self._ensure_root_player(game_state, maximizing_player)
        terminal_value = self._terminal_value(game_state)
        if terminal_value is not None:
            return terminal_value
        if depth == 0:
            return self.evaluator(game_state)

        cache_key = self._cache_key(game_state, maximizing_player)
        cache_entry = self.result_cache.get(cache_key)
        if (
            cache_entry is not None
            and cache_entry["depth"] >= depth
            and cache_entry["flag"] == "exact"
        ):
            return cache_entry["value"]

        candidate_moves = [
            move for move in self._get_ordered_moves(game_state)
            if not move.is_resign
        ]
        if not candidate_moves:
            return self.evaluator(game_state)

        if maximizing_player:
            value = float("-inf")
            for move in candidate_moves:
                child_state = game_state.apply_move(move)
                value = max(
                    value,
                    self.minimax(child_state, depth - 1, False),
                )
        else:
            value = float("inf")
            for move in candidate_moves:
                child_state = game_state.apply_move(move)
                value = min(
                    value,
                    self.minimax(child_state, depth - 1, True),
                )

        self.result_cache.put(cache_key, depth, value, flag="exact")
        return value

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
        self._ensure_root_player(game_state, maximizing_player)
        terminal_value = self._terminal_value(game_state)
        if terminal_value is not None:
            return terminal_value
        if depth == 0:
            return self.evaluator(game_state)

        cache_key = self._cache_key(game_state, maximizing_player)
        cache_entry = self.result_cache.get(cache_key)
        if cache_entry is not None and cache_entry["depth"] >= depth:
            if cache_entry["flag"] == "exact":
                return cache_entry["value"]
            if cache_entry["flag"] == "lower":
                alpha = max(alpha, cache_entry["value"])
            elif cache_entry["flag"] == "upper":
                beta = min(beta, cache_entry["value"])
            if alpha >= beta:
                return cache_entry["value"]

        candidate_moves = [
            move for move in self._get_ordered_moves(game_state)
            if not move.is_resign
        ]
        if not candidate_moves:
            return self.evaluator(game_state)

        original_alpha = alpha
        original_beta = beta

        if maximizing_player:
            value = float("-inf")
            for move in candidate_moves:
                child_state = game_state.apply_move(move)
                value = max(
                    value,
                    self.alphabeta(
                        child_state, depth - 1, alpha, beta, False
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for move in candidate_moves:
                child_state = game_state.apply_move(move)
                value = min(
                    value,
                    self.alphabeta(
                        child_state, depth - 1, alpha, beta, True
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break

        flag = "exact"
        if value <= original_alpha:
            flag = "upper"
        elif value >= original_beta:
            flag = "lower"
        self.result_cache.put(cache_key, depth, value, flag=flag)
        return value

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
        player = self._root_player or game_state.next_player
        score_margin = self._score_margin_for_player(game_state, player)

        my_stones = 0
        opponent_stones = 0
        my_liberties = 0
        opponent_liberties = 0
        my_atari_stones = 0
        opponent_atari_stones = 0
        seen_strings = set()

        center_row = (game_state.board.num_rows + 1) / 2.0
        center_col = (game_state.board.num_cols + 1) / 2.0
        position_score = 0.0

        for row in range(1, game_state.board.num_rows + 1):
            for col in range(1, game_state.board.num_cols + 1):
                point = Point(row, col)
                stone = game_state.board.get(point)
                if stone is None:
                    continue

                distance_to_center = abs(point.row - center_row) + abs(
                    point.col - center_col
                )
                local_bonus = max(
                    0.0, game_state.board.num_rows - distance_to_center
                )
                if stone == player:
                    my_stones += 1
                    position_score += local_bonus
                else:
                    opponent_stones += 1
                    position_score -= local_bonus

                go_string = game_state.board.get_go_string(point)
                if go_string is None:
                    continue
                string_id = id(go_string)
                if string_id in seen_strings:
                    continue
                seen_strings.add(string_id)

                if go_string.color == player:
                    my_liberties += go_string.num_liberties
                    if go_string.num_liberties == 1:
                        my_atari_stones += len(go_string.stones)
                else:
                    opponent_liberties += go_string.num_liberties
                    if go_string.num_liberties == 1:
                        opponent_atari_stones += len(go_string.stones)

        stone_diff = my_stones - opponent_stones
        liberty_diff = my_liberties - opponent_liberties
        atari_diff = opponent_atari_stones - my_atari_stones

        return (
            10.0 * score_margin
            + 1.5 * stone_diff
            + 0.25 * liberty_diff
            + 2.0 * atari_diff
            + 0.1 * position_score
        )

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        moves = [move for move in game_state.legal_moves() if not move.is_resign]
        if not moves:
            return game_state.legal_moves()

        scored_moves = []
        for move in moves:
            score = self._score_move_for_ordering(game_state, move)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]

    def _cache_key(self, game_state, maximizing_player):
        """构造缓存键，区分根视角与轮次信息。"""
        return (
            self._root_player,
            game_state.next_player,
            game_state.board.zobrist_hash(),
            maximizing_player,
        )

    def _ensure_root_player(self, game_state, maximizing_player=None):
        """确保评估函数始终以根节点玩家视角打分。"""
        if self._root_player is not None:
            return
        if maximizing_player is None:
            self._root_player = game_state.next_player
            return
        if maximizing_player:
            self._root_player = game_state.next_player
        else:
            self._root_player = game_state.next_player.other

    def _terminal_value(self, game_state):
        """若已到终局，返回一个绝对值较大的终局分数。"""
        if not game_state.is_over():
            return None

        winner = game_state.winner()
        if winner == self._root_player:
            if game_state.last_move is not None and game_state.last_move.is_resign:
                return 10000.0
            return 10000.0 + self._score_margin_for_player(
                game_state, self._root_player
            )
        if winner == self._root_player.other:
            if game_state.last_move is not None and game_state.last_move.is_resign:
                return -10000.0
            return -10000.0 + self._score_margin_for_player(
                game_state, self._root_player
            )
        return 0.0

    def _score_margin_for_player(self, game_state, player):
        """把计分结果转换为指定玩家视角下的分差。"""
        result = compute_game_result(game_state)
        black_margin = result.b - (result.w + result.komi)
        if player == Player.black:
            return black_margin
        return -black_margin

    def _score_move_for_ordering(self, game_state, move):
        """给候选步打启发式分数，便于 Alpha-Beta 更早剪枝。"""
        if move.is_pass:
            score_margin = self._score_margin_for_player(
                game_state, game_state.next_player
            )
            return -1.0 if score_margin > 0 else 1.0

        next_state = game_state.apply_move(move)
        own_string = next_state.board.get_go_string(move.point)
        liberties = own_string.num_liberties if own_string is not None else 0

        captured_stones = self._count_captured_stones(game_state, next_state)
        adjacent_allies = self._count_adjacent_stones(
            game_state, move, game_state.next_player
        )
        adjacent_opponents = self._count_adjacent_stones(
            game_state, move, game_state.next_player.other
        )

        center_row = (game_state.board.num_rows + 1) / 2.0
        center_col = (game_state.board.num_cols + 1) / 2.0
        center_bias = -0.1 * (
            abs(move.point.row - center_row) + abs(move.point.col - center_col)
        )

        score = (
            5.0 * captured_stones
            + 1.5 * liberties
            + 1.0 * adjacent_allies
            + 0.5 * adjacent_opponents
            + center_bias
        )
        if liberties <= 1:
            score -= 4.0
        if next_state.is_over():
            score += self._terminal_value(next_state) or 0.0
        return score

    def _count_captured_stones(self, game_state, next_state):
        """统计该步对对手造成的提子数。"""
        player = game_state.next_player
        opponent = player.other
        before = self._count_stones(game_state, opponent)
        after = self._count_stones(next_state, opponent)
        return before - after

    def _count_stones(self, game_state, player):
        """统计某一方当前在棋盘上的子数。"""
        total = 0
        for row in range(1, game_state.board.num_rows + 1):
            for col in range(1, game_state.board.num_cols + 1):
                if game_state.board.get(Point(row, col)) == player:
                    total += 1
        return total

    def _count_adjacent_stones(self, game_state, move, player):
        """统计某步周围四邻中指定颜色棋子的数量。"""
        count = 0
        for neighbor in move.point.neighbors():
            if not game_state.board.is_on_grid(neighbor):
                continue
            if game_state.board.get(neighbor) == player:
                count += 1
        return count



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
        existing = self.cache.get(zobrist_hash)
        new_entry = {
            "depth": depth,
            "value": value,
            "flag": flag,
        }

        if existing is None:
            self.cache[zobrist_hash] = new_entry
            return

        if depth > existing["depth"]:
            self.cache[zobrist_hash] = new_entry
            return

        if depth == existing["depth"]:
            if existing["flag"] != "exact" and flag == "exact":
                self.cache[zobrist_hash] = new_entry
            elif existing["flag"] == flag:
                self.cache[zobrist_hash] = new_entry
