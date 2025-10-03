# Evolutionary training loop


import game
from network import model
import torch
import random
from timer import Timer
from network_tools import random_board

population_size = 10
food_size = 3 #Number of boards/games each generation will play.
timer = Timer()

def make_initial_population(pop_size, model_class):
    """
    Purpose:
        Create an initial population of models.
    Input:
        pop_size: number of models to create
        model_class: the model
    Output:
        A dictionary:
            keys: model instances
            values: 0 (initial fitness score)
    """
    torch.manual_seed(42)

    timer.start()
    population = {model_class():0 for i in range(pop_size)} 
    timer.print_time("Initial population created.")

    return population

def select_two_models(population):
    """
    Purpose:
        An iterator to select two models at random from the population, without replacment.
    Input:
        population: dictionary of models
    Output:
        An iterator of sorts, intended to be used in for loops such as:
            for model1, model2 in select_two_models(population):
            #Do stuff with model1 and model2
    """
    models = list(population.keys())
    random.shuffle(models)
    for i in range(0, len(models) - 1, 2):
        yield models[i], models[i + 1]

def generate_food(food_size):
    """
    Purpose:
        Generate the boards/games that the generation will play.
    Input:
        food_size: The number of boards/games to generate.
    Output:
        A list of size food_size, each element a random board.
    """
    generation_food = [random_board() for _ in range(food_size)]
    return generation_food
    

def create_new_generation(old_population, generation_food):
    """
    Purpose:
        Create a new generation.
    Input:
        old_population: A population of models. Dictionary {model: fitness}
        generation_food: The games to play. list of boards, which present is a list of lists. 
    Output:
        A new population. (Currently hardcoded with player 2 always winning by 280 to 200)
    """
    for model1, model2 in select_two_models(pop):
        print(f"Selected models: {pop[model1]} vs {pop[model2]}")
        for board in generation_food:
            scores = [200, 280] #Temp just to be able to run
            old_population[model1] += scores[0]
            old_population[model2] += scores[1]
            print_population(old_population)
            continue
            scores = game.play_game(model1, model2, board) #Does not exist. Need something like this to get them to play against each other.
    print("Population fitness has been updated.")
    print_population(old_population)

#Just for debugging
def print_population(population):
    for i, (model, fitness) in enumerate(population.items()):
        print(f"Model {i}: Fitness: {fitness}")

#Proof of concept:
pop = make_initial_population(population_size, model)
timer.start()
for model1, model2 in select_two_models(pop):
    print(f"Selected models: {pop[model1]} vs {pop[model2]}")
timer.print_time("Iteratated population in pairs.")

create_new_generation(pop, generate_food(food_size))