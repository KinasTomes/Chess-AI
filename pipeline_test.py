import torch
from training.timer import Timer
from core.model import ChessNet
from pipeline import self_play
from training.utils import load_model
from training.replay_buffer import ReplayBuffer
from multiprocessing import freeze_support
import traceback

def main():
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load model
        model = ChessNet()
        try:
            model = load_model(r'model_checkpoint\best_model.pth', model)
            model = model.to(device)  # Move model to device
            model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(traceback.format_exc())
            return

        # Create replay buffer
        replay_buffer = ReplayBuffer()

        # Run self-play
        try:
            self_play(
                model=model,
                num_games=1,
                replay_buffer=replay_buffer,
            )
            print("✅ Self-play completed successfully")
        except Exception as e:
            print(f"❌ Error during self-play: {str(e)}")
            print(traceback.format_exc())

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(traceback.format_exc())

if __name__ == '__main__':
    freeze_support()
    timer = Timer()
    timer.start()
    main()
    timer.end()