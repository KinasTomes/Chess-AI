import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        if self.start_time is not None:
            print("⏳ The timer has already started!")
            return
        
        self.start_time = time.time()
        print("🕰️ Timer started...")

    def end(self):
        if self.start_time is None:
            print("❌ The timer hasn't started!")
            return
        
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000
        print(f"⏱️ Time elapsed: {elapsed_time:.3f} ms")
        self.start_time = None
        self.end_time = None