# Mancala game logic
class Mancala:
    def __init__(self, board=None):
        # Board layout:
        # Indexes 0–5: Player 1 pits, 6: Player 1 store
        # Indexes 7–12: Player 2 pits, 13: Player 2 store
        if board != None:
            self.board = board
        else:
            self.board = [4] * 6 + [0] + [4] * 6 + [0]
        self.current_player = 1

    def get_legal_moves(self):
        return ([i for i in range(6) if self.board[i] > 0]
                if self.current_player == 1
                else [i for i in range(7, 13) if self.board[i] > 0])

    def is_game_over(self):
        return (all(self.board[i] == 0 for i in range(6)) or
                all(self.board[i] == 0 for i in range(7, 13)))

    def _get_opposite_index(self, index):
        return 12 - index

    def _sweep_remaining_stones(self):
        # Sweep Player 1 pits into store
        for i in range(6):
            self.board[6] += self.board[i]
            self.board[i] = 0
        # Sweep Player 2 pits into store
        for i in range(7, 13):
            self.board[13] += self.board[i]
            self.board[i] = 0

    def apply_move(self, pit_index):
        if self.board[pit_index] == 0:
            raise ValueError("Cannot move from an empty pit")

        stones = self.board[pit_index]
        self.board[pit_index] = 0
        index = pit_index
        last_index = -1  # Track where last stone lands

        while stones > 0:
            index = (index + 1) % 14
            # Skip opponent's store
            if self.current_player == 1 and index == 13:
                continue
            if self.current_player == 2 and index == 6:
                continue
            stones -= 1
            self.board[index] += 1
            last_index = index

        # --- Capture Logic ---
        if self.current_player == 1 and 0 <= last_index <= 5:
            if self.board[last_index] == 1:  # last stone landed in empty pit
                opposite_index = self._get_opposite_index(last_index)
                captured = self.board[opposite_index]
                if captured > 0:
                    self.board[6] += captured + 1
                    self.board[last_index] = 0
                    self.board[opposite_index] = 0

        elif self.current_player == 2 and 7 <= last_index <= 12:
            if self.board[last_index] == 1:
                opposite_index = self._get_opposite_index(last_index)
                captured = self.board[opposite_index]
                if captured > 0:
                    self.board[13] += captured + 1
                    self.board[last_index] = 0
                    self.board[opposite_index] = 0

        # --- Extra turn ---
        if (self.current_player == 1 and last_index == 6) or (self.current_player == 2 and last_index == 13):
            if self.is_game_over():
                self._sweep_remaining_stones()
            return  # Player keeps the turn

        # --- Switch player ---
        self.current_player = 2 if self.current_player == 1 else 1

        if self.is_game_over():
            self._sweep_remaining_stones()
        
    # model1(board) yields the move model1 chooses. 
    def play_game(model1, model2):
        pass

    def get_score(self):
        return (self.board[6], self.board[13])

    def print_board(self):
        print("Player 2 side: " + " ".join(map(str, self.board[12:6:-1])))
        print(f"Stores => P1: {self.board[6]} | P2: {self.board[13]}")
        print("Player 1 side: " + " ".join(map(str, self.board[0:6])))

'''
	•	__init__: Sets up the board and starting player.
	•	get_legal_moves(): Returns valid pit indexes for the current player.
	•	is_game_over(): Checks if either side is empty.
	•	apply_move(pit_index): Performs a move and handles captures, extra turns, and turn switching.
	•	get_score(): Returns the current score and sweeps remaining stones if the game is over.
	•	print_board(): Displays the board in a human-readable format.
'''
