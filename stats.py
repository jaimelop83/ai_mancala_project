class Stats:
    def __init__(self):
        self.total_games = 0
        self.total_moves = 0
        self.illegal_moves = 0
        self.p1_wins = 0
        self.p2_wins = 0
        self.ties = 0

    def snapshot(self):
        return {
            "games": self.total_games,
            "moves": self.total_moves,
            "illegal_moves": self.illegal_moves,
            "p1_wins": self.p1_wins,
            "p2_wins": self.p2_wins,
            "ties": self.ties,
        }

GLOBAL_STATS = Stats()

def print_stats(prefix="Run stats"):
    s = GLOBAL_STATS.snapshot()
    print(
        f"{prefix}: "
        f"games={s['games']}  moves={s['moves']}  illegal={s['illegal_moves']}  "
        f"p1_wins={s['p1_wins']}  p2_wins={s['p2_wins']}  ties={s['ties']}"
    )
