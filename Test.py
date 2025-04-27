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

move = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6', 'e1g1', 'f8e7']

# for m in move:
#     obs, _ = env.step(chess.Move.from_uci(m))
#     print(obs.shape)

np.set_printoptions(threshold=np.inf)

obs, _ = env.step(chess.Move.from_uci('e2e4'))
print(obs[0:6])
print(env.board)