import time

"""
Purpose:
    This class doesn't serve any functional purpose.
    I just want to be able to keep track of how long things take, so that we don't accidentally try to optimize something that is too fast to matter.
"""

class Timer:
    def __init__(self):
        self.start_time = None
            
    def start(self):
        self.start_time = time.time()

    def print_time(self, message, start="", end="\n"):
        total = time.time() - self.start_time
        print(f"{start}[{total:05.2f} {message}]", end=end)