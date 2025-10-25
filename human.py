import torch
import os

from  game import Mancala
#from three_layer_model import three_layer_model as model
#from base_single_layer_model import base_single_layer_model as model
from network import PERCENT_RANDOM
#from two_layer_model import two_layer_model as model
from updated_two_layer_model import updateded_two_layer_model as model

game = Mancala()

model_file = os.path.join("models", "55er.pt")
AI = model()
AI.load_state_dict(torch.load(model_file))

while(not game.is_game_over()):
    game.print_board()
    # if game.current_player == 1:
    #     print("Your move!")
    #     move = int(input())
    #     game.apply_move(move)
    # else:
    #     # move = AI.make_move(game.board)
    #     temp_board = [game.board[(i+7)%14] for i in range(14)]
    #     move = AI.make_move(temp_board)
    #     print(f"The AI chose to go {move+7}")
    #     game.apply_move(move+7)
    if game.current_player == 2:
        print("Your move!")
        move = int(input())
        game.apply_move(move)
    else:
        # move = AI.make_move(game.board)
        move = AI.make_move(game.board)
        print(f"The AI chose to go {move}")
        game.apply_move(move)




print("Game over!")
print(game.get_score())

