import ray
import chess

from core.coords_converter import ChessCoordsConverter
from utils.stockfish_worker import StockfishWorker, parallel_evaluate_legal_moves

stockfish_path = r'evaluation\stockfish\stockfish-windows-x86-64-avx2.exe'
cvt = ChessCoordsConverter()

ray.init(num_cpus=8)
num_workers = 8
workers = [StockfishWorker.remote(stockfish_path) for _ in range(num_workers)]

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
board = chess.Board(fen)

policy = parallel_evaluate_legal_moves(board, workers, time_limit=5)
policy = {move: score for move, score in policy.items() if score > 0}
print(policy)