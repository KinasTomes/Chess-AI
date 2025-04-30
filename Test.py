# import ray
# import chess

# from core.coords_converter import ChessCoordsConverter
# from utils.stockfish_worker import StockfishWorker, parallel_evaluate_legal_moves

# stockfish_path = r'evaluation\stockfish\stockfish-windows-x86-64-avx2.exe'
# cvt = ChessCoordsConverter()

# ray.init(num_cpus=8)
# num_workers = 8
# workers = [StockfishWorker.remote(stockfish_path) for _ in range(num_workers)]

# fen = 'r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQ - 0 1'
# board = chess.Board(fen)

# policy = parallel_evaluate_legal_moves(board, workers, time_limit=50)
# for move, score in policy.items():
#     print(f"Move: {move}, Score: {score:.4f}")

import chess
import numpy as np
from core.chess_base_v2 import ChessEnv

env = ChessEnv()

print(env.board.result())