import os
import chess
import chess.pgn
import chess.engine
import numpy as np
import ray
import torch

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
        while game_idx < 2000:
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

def process_pgn_directory_to_buffer(directory_path: str, save_dir: str = "buffer_data", samples_per_file: int = 10000) -> None:
    """
    Xử lý tất cả các file PGN trong thư mục và lưu thành các file buffer.

    Args:
        directory_path: Đường dẫn đến thư mục chứa các file PGN
        save_dir: Thư mục để lưu các file buffer
        samples_per_file: Số samples tối đa trong mỗi file buffer
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Lấy danh sách các file PGN
    pgn_files = [f for f in os.listdir(directory_path) if f.endswith('.pgn')]
    print(f"🎯 Tìm thấy {len(pgn_files)} file PGN trong thư mục {directory_path}")
    
    # Khởi tạo Ray và các worker
    ray.init(num_cpus=8)
    num_workers = 8
    workers = [StockfishWorker.remote(stockfish_path) for _ in range(num_workers)]
    
    try:
        # Buffer tạm thời để gom samples
        temp_buffer = ReplayBuffer()
        current_part = 0
        
        # Xử lý từng file PGN
        for pgn_file in pgn_files:
            print(f"\n📂 Đang xử lý file: {pgn_file}")
            pgn_path = os.path.join(directory_path, pgn_file)
            
            # Đọc file PGN
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    print(f"❌ Không đọc được ván cờ từ file {pgn_path}")
                    continue

                board = game.board()
                moves = list(game.mainline_moves())
                
                # Xác định người chơi
                w, b = (1, 2) if board.turn else (2, 1)
                env = ChessEnv(white_player_id=w, black_player_id=b)
                game_history = []

                # Xử lý từng nước đi
                for move in moves:
                    # Đánh giá các nước đi hợp lệ
                    scores = parallel_evaluate_legal_moves(env.chess_board, workers)
                    
                    # Lưu trạng thái và policy
                    game_history.append({
                        'state': env._observation(),
                        'policy': map_legal_move_scores_to_policy(scores),
                        'player': env.to_play
                    })
                    
                    # Thực hiện nước đi
                    env.step(env.chess_coords.move_to_index(move))

                # Xác định kết quả ván đấu
                if env.winner == env.white_player:
                    game_result = 1
                elif env.winner == env.black_player:
                    game_result = -1    
                else:
                    game_result = 0

                # Thêm vào buffer
                temp_game_data = []
                for history in game_history:
                    if history['player'] == env.white_player:
                        value = game_result
                    else:
                        value = -game_result

                    temp_game_data.append((history['state'], history['policy'], value))
                    
                temp_buffer.add_game(temp_game_data)
                # Kiểm tra xem có đủ samples để lưu file mới không
                if len(temp_buffer) >= samples_per_file:
                    # Lấy samples_per_file samples đầu tiên
                    states, policies, values = temp_buffer.sample_batch(samples_per_file)
                    
                    # Lưu thành file riêng
                    save_path = os.path.join(save_dir, f"buffer_part_{current_part}.pt")
                    torch.save({
                        'states': states,
                        'policies': policies,
                        'values': values
                    }, save_path)
                    print(f"💾 Đã lưu file {save_path} với {len(states)} samples")
                    
                    # Xóa samples đã lưu khỏi buffer
                    temp_buffer.clear(10000)
                    current_part += 1
        
        # Lưu phần samples còn lại nếu có
        if len(temp_buffer) > 0:
            states, policies, values = temp_buffer.sample_batch(len(temp_buffer))
            save_path = os.path.join(save_dir, f"buffer_part_{current_part}.pt")
            torch.save({
                'states': states,
                'policies': policies,
                'values': values
            }, save_path)
            print(f"💾 Đã lưu file {save_path} với {len(states)} samples")
            
    finally:
        # Tắt các worker
        for worker in workers:
            worker.shutdown.remote()
        # Tắt Ray
        ray.shutdown()

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    
    # Xử lý tất cả các file PGN trong thư mục
    process_pgn_directory_to_buffer(
        directory_path="sample_data/fen_database",
        save_dir="buffer_data",
        samples_per_file=10000
    )
    
    timer.end()