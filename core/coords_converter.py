import chess

class ChessCoordsConverter:
    def __init__(self):
        # Directions for queen-like moves (8 hướng)
        self.queen_dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Straight
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonal
        ]

        # Knight moves (8 hướng)
        self.knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (-1, 2), (1, -2), (-1, -2)
        ]

        # promotion options (3 hướng × 4 loại)
        self.promotions = [
            (chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN),  # Thẳng
            (chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN),  # Chéo trái
            (chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN)   # Chéo phải
        ]

    def move_to_index(self, move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)
        dr, df = to_rank - from_rank, to_file - from_file

        # Knight moves (512 indices: 3584-4095)
        if (dr, df) in self.knight_moves:
            return 3584 + from_sq * 8 + self.knight_moves.index((dr, df))

        # promotions (768 indices: 4096-4863)
        if move.promotion:
            promo_dir = None
            # Xác định hướng phong cấp
            if from_rank == 6 and to_rank == 7:  # Trắng
                if df == 0: promo_dir = 0
                elif df == -1: promo_dir = 1
                elif df == 1: promo_dir = 2
            elif from_rank == 1 and to_rank == 0:  # Đen
                if df == 0: promo_dir = 0
                elif df == 1: promo_dir = 1
                elif df == -1: promo_dir = 2

            if promo_dir is not None:
                try:
                    promo_type = self.promotions[promo_dir].index(move.promotion)
                    return 4096 + from_sq * 12 + promo_dir * 4 + promo_type
                except ValueError:
                    raise ValueError(f"Invalid promotion type: {move.promotion}")

        # Queen moves (3584 indices: 0-3583)
        for dir_idx, (dx, dy) in enumerate(self.queen_dirs):
            for steps in range(1, 8):
                if (from_rank + dx*steps == to_rank and
                        from_file + dy*steps == to_file):
                    idx = from_sq * 56 + dir_idx * 7 + (steps - 1)
                    return idx

        raise ValueError(f"Invalid move: {move.uci()}")

    def index_to_move(self, index: int) -> chess.Move:
        # Queen moves (0-3583)
        if index < 3584:
            from_sq = index // 56
            rem = index % 56
            dir_idx = rem // 7
            steps = rem % 7 + 1
            dx, dy = self.queen_dirs[dir_idx]
            from_rank = from_sq // 8
            from_file = from_sq % 8
            to_rank = from_rank + dx * steps
            to_file = from_file + dy * steps
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                return chess.Move(from_sq, to_sq)
            else:
                raise ValueError("Invalid queen move index")

        # Knight moves (3584-4095)
        elif index < 4096:
            index -= 3584
            from_sq = index // 8
            move_idx = index % 8
            dx, dy = self.knight_moves[move_idx]
            from_rank = from_sq // 8
            from_file = from_sq % 8
            to_rank = from_rank + dx
            to_file = from_file + dy
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                return chess.Move(from_sq, to_sq)
            else:
                raise ValueError("Invalid knight move index")

        # promotions (4096-4863)
        elif index < 4864:
            index -= 4096
            from_sq = index // 12
            rem = index % 12
            promo_dir = rem // 4
            promo_type = rem % 4

            from_rank = from_sq // 8
            from_file = from_sq % 8
            if from_rank == 6:  # Trắng
                directions = [(1, 0), (1, -1), (1, 1)]
            elif from_rank == 1:  # Đen
                directions = [(-1, 0), (-1, 1), (-1, -1)]
            else:
                raise ValueError("Invalid rank for promotion")

            dx, dy = directions[promo_dir]
            to_rank = from_rank + dx
            to_file = from_file + dy
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                return chess.Move(
                    from_sq,
                    to_sq,
                    promotion=self.promotions[promo_dir][promo_type]
                )
            else:
                raise ValueError("Invalid promotion move index")

        else:
            raise ValueError("Invalid index")