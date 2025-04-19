import os
import chess
import chess.pgn
import chess.engine
import numpy as np
import ray

from typing import Dict
from training.timer import Timer
from core.chess_base import ChessEnv
from training.replay_buffer import ReplayBuffer
from core.coords_converter import ChessCoordsConverter
from utils.stockfish_worker import StockfishWorker, parallel_evaluate_legal_moves

stockfish_path = r'evaluation\stockfish\stockfish-windows-x86-64-avx2.exe'

def split_pgn_file(input_pgn_path: str, output_dir: str = None) -> None:
    """
    Tách một file PGN lớn thành các file PGN nhỏ, mỗi file chứa một ván cờ.

    Args:
        input_pgn_path (str): Đường dẫn đến file PGN gốc.
        output_dir (str): Thư mục để lưu các file PGN nhỏ.
    """
    if not output_dir is None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_pgn_path, encoding='utf-8') as pgn_file:
        game_idx = 1
        while game_idx < 1000:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            if output_dir is None:
                output_path = f"game_{game_idx}.pgn"
            else:
                output_path = os.path.join(output_dir, f"game_{game_idx}.pgn")

            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                output_file.write(game.accept(exporter))
            print(f"✅ Saved game {game_idx} to {output_path}")
            game_idx += 1

    print(f"\n🎯 Tổng cộng {game_idx - 1} ván đã được tách và lưu trong thư mục '{output_dir}'.")

def get_fen_and_moves_from_pgn_file(pgn_path: str) -> tuple[list[str], list[str]]:
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            print("❌ Không đọc được ván cờ.")
            return [], []

        board = game.board()
        fen_list = []
        uci_moves = []

        for move in game.mainline_moves():
            fen_list.append(board.fen())
            uci_moves.append(move.uci())
            board.push(move)

        return fen_list, uci_moves
    
def map_legal_move_scores_to_policy(legal_moves: Dict[str, float], action_dim: int = 4864) -> np.ndarray:
    policy = np.zeros(action_dim, dtype=np.float32)
    converter = ChessCoordsConverter()
    for move_uci, score in legal_moves.items():
        move_index = converter.move_to_index(chess.Move.from_uci(move_uci))
        policy[move_index] = score
    return policy
    
def get_replay_buffer_from_pgn(pgn_path: str) -> ReplayBuffer:
    # Khởi tạo Ray
    ray.init(num_cpus=8)
    
    # Khởi tạo các worker
    num_workers = 8  # Số worker bằng số CPU cores
    workers = [StockfishWorker.remote(stockfish_path) for _ in range(num_workers)]
    
    try:
        fens, moves = get_fen_and_moves_from_pgn_file(pgn_path)
        w, b = (1, 2) if chess.Board(fens[0]).turn else (2, 1)
        env = ChessEnv(white_player_id=w, black_player_id=b)
        replay_buffer = ReplayBuffer()
        game_history = []

        for move in moves:
            scores = parallel_evaluate_legal_moves(env.chess_board, workers)
            game_history.append({
                'state': env._observation(),
                'policy': map_legal_move_scores_to_policy(scores),
                'player': env.to_play
            })
            env.step(env.chess_coords.move_to_index(chess.Move.from_uci(move)))

        if env.winner == env.white_player:
            game_result = 1
        elif env.winner == env.black_player:
            game_result = -1    
        else:
            game_result = 0

        for history in game_history:
            if history['player'] == env.white_player:
                value = game_result
            else:
                value = -game_result

            replay_buffer.add_game([(history['state'], history['policy'], value)])
        
        return replay_buffer
        
    finally:
        # Tắt các worker
        for worker in workers:
            worker.shutdown.remote()
        # Tắt Ray
        ray.shutdown()

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    rb = get_replay_buffer_from_pgn(r"sample_data\ficsgame_2024_chess2000\game_1.pgn")
    timer.end()