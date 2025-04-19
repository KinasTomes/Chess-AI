import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=500000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_data):
        self.buffer.extend(game_data)

    def sample_batch(self, batch_size):
        batch = list(self.buffer)[:batch_size]
        states, policies, values = zip(*batch)
        return (
            np.stack(states),           # (B, 119, 8, 8)
            np.stack(policies),        # (B, action_dim)
            np.array(values, dtype=np.float32)  # (B,)
        )

    def clear(self, sample_size: int):
        for _ in range(sample_size):
            if self.buffer:
                self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)