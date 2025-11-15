import torch
from collections import Counter
from mancala_model import mancala_model
import os

class ensemble_model(mancala_model):
    model_name = "Ensemble"

    def __init__(self):
        super().__init__()

        self.models = []
        from updated_fancy_model import updated_fancy_model 
        AI = updated_fancy_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "fancy.pt")))
        self.models.append(AI)
        from updated_transformer_model import updated_transformer_model 
        AI = updated_transformer_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "transformer.pt")))
        self.models.append(AI)
        from updated_cnn_model import updated_cnn_model 
        AI = updated_cnn_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "cnn.pt")))
        self.models.append(AI)
        from updated_three_layer_model import updated_three_layer_model 
        AI = updated_three_layer_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "3_layer.pt")))
        self.models.append(AI)
        from updated_two_layer_model import updated_two_layer_model 
        AI = updated_two_layer_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "2_layer.pt")))
        self.models.append(AI)
        from updated_single_layer_model import updated_single_layer_model 
        AI = updated_single_layer_model()
        AI.load_state_dict(torch.load(os.path.join("models", "Best", "1_layer.pt")))
        self.models.append(AI)

        self.fitness = 0
        self.special = False
        self.eval()

    def forward(self, x):
        votes = []
        for model in self.models:
            move = model(x).item()
            votes.append(move)
        # print(f"{votes=}")

        counts = Counter(votes)
        most_common_votes = torch.tensor(counts.most_common()[0][0])
        print(f"{most_common_votes=}")

        if len(set(votes)) == 6:
            print(f"Short circut to fancy, {votes[0]=}")
            return torch.tensor(votes[0][0]) #fancy model if they're evenly split
        
        return most_common_votes
