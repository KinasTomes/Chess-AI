import torch
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
    
    def take_all(self):
        batch = list(self.buffer)
        states, policies, values = zip(*batch); del batch
        return (
            torch.from_numpy(np.stack(states)).float(),  # (B, 119, 8, 8)
            torch.from_numpy(np.stack(policies)).float(),  # (B, action_dim)
            torch.from_numpy(np.array(values, dtype=np.float32))  # (B,)
        )
    
    def clear(self, num: int) -> None:
        if len(self.buffer) >= num:
            self.buffer = deque(list(self.buffer)[num:], maxlen=self.max_size)

    def __len__(self):
        return len(self.buffer)