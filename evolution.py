# Evolutionary training loop


from network import model
import torch
import random
from timer import Timer

population_size = 10
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
            values: None (initial fitness score)
    """
    torch.manual_seed(42)

    timer.start()
    population = {model_class():i for i in range(pop_size)} #NOTE: Change i to None. This is just for debugging so I can see where they're really going.
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

def create_new_generation(old_population):
    pass

#Proof of concept:
pop = make_initial_population(population_size, model)
timer.start()
for model1, model2 in select_two_models(pop):
    print(f"Selected models: {pop[model1]}, {pop[model2]}")
timer.print_time("Iteratated population in pairs.")

