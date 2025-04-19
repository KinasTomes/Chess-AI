import ray
import numpy as np
import chess.engine

from typing import Dict

@ray.remote
class StockfishWorker:
    def __init__(self, stockfish_path: str, time_limit: float = 0.1):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.time_limit = time_limit

    def evaluate(self, fen: str, move_uci: str) -> tuple[str, float]:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        try:
            info = self.engine.analyse(board, chess.engine.Limit(time=self.time_limit))
            score = info['score'].white().score(mate_score=1000000)
            if board.turn:
                score = -score
            return move_uci, score / 100 if score is not None else 0
        except:
            return move_uci, 0

    def shutdown(self):
        self.engine.quit()

def normalize_score(scores: Dict[str, float]) -> Dict[str, float]:
    score_values = np.array(list(scores.values()))
    exp_scores = np.exp(score_values - np.max(score_values))
    normalized_score = exp_scores / np.sum(exp_scores)
    return dict(zip(scores.keys(), normalized_score))

def parallel_evaluate_legal_moves(board: chess.Board, workers: list, time_limit: float = 0.1) -> Dict[str, float]:
    fen, legal_moves = board.fen(), list(board.legal_moves)
    move_args = [(fen, move.uci()) for move in legal_moves]

    futures = []
    for i, (fen, move_uci) in enumerate(move_args):
        worker = workers[i % len(workers)]  # Phân chia đều vào các worker
        futures.append(worker.evaluate.remote(fen, move_uci))

    results = ray.get(futures)
    move_scores = {move_uci: score for move_uci, score in results}
    return normalize_score(move_scores)
