from core.chess_base import ChessEnv
from core.coords_converter import ChessCoordsConverter
from core.mcts import MCTS
from core.model import ChessNet
from training.utils import load_predict_model
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    env = ChessEnv()
    cvt = ChessCoordsConverter()

    fen = 'r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQ - 0 1'
    env.chess_board.set_fen(fen)

    env._update_board_from_chess_board()
    env._update_captures()
    env._update_game_state()
    env._update_legal_actions()

    env.render()

    model = ChessNet()
    model = load_predict_model(r'model_checkpoint\best_model.pth', model)
    model.eval()

    mcts = MCTS(
        neural_net=model,
        converter=cvt,
        env=env,
        simulations=500,
        max_depth=10,
        device='cuda',
        num_processes=4,
        use_model=True 
    )

    pi = mcts.run(env.chess_board)
    for i, prob in enumerate(pi):
        if prob != 0:
            print(f"Move: {cvt.index_to_move(i)}, Probability: {prob:.4f}")
    print("Total probability:", sum(pi))
    print("MCTS simulation completed.")