import time

"""
Purpose:
    This class doesn't serve any functional purpose.
    I just want to be able to keep track of how long things take, so that we don't accidentally try to optimize something that is too fast to matter.
"""

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
            
    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.start_time = None

    def print_time(self, message):
        self.stop()
        total = self.elapsed
        if self.start_time is not None:
            total += time.time() - self.start_time
        print(f"{message}: {total:.4f} seconds")