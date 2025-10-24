import random

# Mancala game logic
class Mancala:
    def __init__(self, board=None):
        # Board layout:
        # Indexes 0–5: Player 1 pits, 6: Player 1 store
        # Indexes 7–12: Player 2 pits, 13: Player 2 store
        if board is not None:
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
            #print(f"Added stone to {index}")

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
    def play_game(self, model1, model2):
        #print("Playing game!")
        #print(f"{torch.tensor(self.board).unsqueeze(0)=}")
        #move = model1(torch.tensor(self.board).unsqueeze(0))
        #final_score = [None, 0,0] #1-indexed
        number_of_moves = 0

        while True:
            if self.current_player == 1:
                move = model1.make_move(self.board)
            else:
                temp_board = [self.board[(i+7)%14] for i in range(14)]
                move = model2.make_move(temp_board)
                move = move + 7
            if move not in self.get_legal_moves():
                #print(f"\t[{number_of_moves}]Game is over with player {self.current_player} making an illegal move: {move}")
                #final_score[1] = self.get_score()[0] * 10
                #final_score[1] = self.get_score()[1] * 10
                #final_score[self.current_player] -= 10000
                self.board[7*self.current_player-1] += -1000  # f(x)=7x-1 gives 1->6 & 2->13.
                #final_score[3  - self.current_player] += self.get_score()[2  - self.current_player] * 10
                break
            #print(f"\t[{number_of_moves}] Player {self.current_player} makes move {move}")
            self.apply_move(move)
            #self.print_board()
            if self.is_game_over():
                #print(f"\t[{number_of_moves}] Game is over - with a WINNER!")
                #final_score[1] = self.get_score()[0] * 10
                #final_score[1] = self.get_score()[1] * 10
                break
            number_of_moves += 1
            if number_of_moves > 100:
                print("Something went wrong, 100 moves is probably impossible")
                assert(False)
            #print()
        #self.print_board()
        return
        #self.print_board()

    def get_score(self):

        #Stand-in code so other things are operational
        # r = random.random()
        # a = random.randrange(0, 481, 10)
        # b = 480 - a
        # if r < 0.25: return (-10000,a)
        # if r < 0.5: return (b,-10000)
        # elif r < 0.52: return (10000+max(a,b), min(a,b))
        # elif r < 0.52: return (min(a,b), 10000+max(a,b))
        # return (a, b)

        p1_score = self.board[6] * 10
        p2_score = self.board[13] * 10
        
        #Game finished with an illegal move, one bin recieved a -1000. Do not award winning points.
        if p1_score < 0 or p2_score < 0:
             pass
        elif p1_score == 0 and p2_score == 0: #Total shut out no bonus points
            pass
        elif p1_score > p2_score: #Award winning bonus  points
            p1_score += 10000
            p2_score += 1000
        elif p2_score > p1_score: #Award winning bonus points
            p2_score += 10000
            p1_score += 1000
        elif p1_score == p2_score: #Split winning bonus points
            p1_score += 10000/2
            p2_score += 10000/2
        #print(f"{self.board=}")
        #print(f"\t\tScores reported: {p1_score}, {p2_score}")

        return (p1_score, p2_score)

    def print_board(self):
        print("\t\tPlayer 2 side: " + " ".join(map(str, self.board[12:6:-1])))
        print(f"\t\tStores => P1: {self.board[6]} | P2: {self.board[13]}")
        print("\t\tPlayer 1 side: " + " ".join(map(str, self.board[0:6])))

'''
	•	__init__: Sets up the board and starting player.
	•	get_legal_moves(): Returns valid pit indexes for the current player.
	•	is_game_over(): Checks if either side is empty.
	•	apply_move(pit_index): Performs a move and handles captures, extra turns, and turn switching.
	•	get_score(): Returns the current score and sweeps remaining stones if the game is over.
	•	print_board(): Displays the board in a human-readable format.
'''

#from network import model
# from base_single_layer_model import base_single_layer_model as model
# model1 = model()
# model2 = model()

#testo = Mancala([5,1,5,5,5,0,1, 5, 1, 0, 7, 6, 6, 2])
#testo.print_board()
#testo.apply_move(0)
#testo.print_board()
# final_score = testo.play_game(model1, model2)
# print(f"{final_score=}")


