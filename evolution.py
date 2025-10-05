# Evolutionary training loop


import game
from network import model
import torch
import random
from timer import Timer
from network import *
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
        A list of models
    """
    timer.start()
    population = [model_class() for i in range(pop_size)]
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
    random.shuffle(population)
    for i in range(0, len(population) - 1, 2):
        yield population[i], population[i + 1]

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
    

def survival_of_the_fittest(population, generation_food):
    """
    Purpose:
        Feed the old population and updates its fitness.
    Input:
        old_population: A population of models
        generation_food: The games to play. list of boards, which present is a list of lists. 
    Output:
        a 'new' population, now with updated fitness scores!)
    """
    for model1, model2 in select_two_models(pop):
        #print(f"Selected models: {pop[model1]} vs {pop[model2]}")
        for board in generation_food:
            scores = [200, 280] #Temp just to be able to run
            model1.fitness += scores[0]
            model2.fitness += scores[1]
            #print_population(old_population)
            continue #TODO need line below, possibly will modification
            scores = game.play_game(model1, model2, board) #Does not exist. Need something like this to get them to play against each other.
    print("Population fitness has been updated.")
    print_population(population)
    return population

def sort_population(population):
    """
    Purpose:
        Sort the population
    Input:
        population
    Output:
        population, sorted
    """
    sorted_population = sorted(population, key=lambda model: model.fitness, reverse=True)
    return sorted_population

#Just for debugging
def print_population(population):
    for i, model in enumerate(population):
        print(f"Model {i}: Fitness: {model.fitness}")

def reproduce_pair(model1, model2):
    """
    Purpose:
        Create a new model by combining weights from two parent models.
    Input:
        model1, model2: parent models (same class)
        model1_similarity: percent of parameters to take from model1
    Output:
        A new child model with mixed weights.
    """
    global model_similarity

    #Get the parameters from the parents. A bit of machinery to flatten them. 
    params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
    number_of_parameters = params1.numel()

    #Make a child.
    child = type(model1)()
    child_params = [0.]*number_of_parameters

    #Figure out what parameters come from which parent.
    indices = torch.randperm(number_of_parameters)
    split = int(MODEL_SIMILARITY * number_of_parameters)
    parent1_idx = indices[:split]
    parent2_idx = indices[split:]

    #Reproduce
    child_flat = params1.clone()
    child_flat[parent2_idx] = params2[parent2_idx]

    # Populate child weights how??
    pointer = 0
    for p in child.parameters():
        num_params = p.numel()
        # Copy the correct slice back into parameter shape
        p.data = child_flat[pointer:pointer + num_params].view_as(p).clone()
        pointer += num_params

    
    if False: #Debugging code to make sure the child is actually a mix of the parents.
        params_child = torch.cat([p.data.view(-1) for p in child.parameters()])
        p1_count = 0
        p2_count = 0
        for p1,p2,c in zip(params1, params2, params_child):
            if not (p1 == c or p2 == c):
                print(f"Error: Child parameter {c} does not match either parent {p1} or {p2}.")
                assert(False)
            if p1 ==c:
                p1_count += 1
                print(".", end="")
            if p2 == c:
                p2_count += 1
                print("*", end="")
        tot = p1_count + p2_count
        print(f"{indices=}")
        print(f"\n{p1_count=} {p2_count=} {tot=}")
        print(f"{split=}")
        assert(False)

    return child

def reproduce_pop(population):
    """
    Purpose:
        Create a new generation.
    Input:
        population (Note that it need not be sorted)
    Output:
        population, with the bottom network.DEATH_RATE members replaced by children from the above network.reproductive_floor.
    """
    global DEATH_RATE, REPRODUCTIVE_FLOOR
    population = sort_population(population)

    weakest_fit_to_reproduce = int(REPRODUCTIVE_FLOOR * len(population))

    weakest_surviving_model = int(DEATH_RATE * len(population))
    for i in range(-weakest_surviving_model, 0):
        parent_indices = random.sample(range(weakest_fit_to_reproduce), 2)
        new_child_member = reproduce_pair(population[parent_indices[0]], population[parent_indices[1]] )
        population[i] = new_child_member

    return population
    
    
    

#Proof of concept:
pop = make_initial_population(population_size, model)
# timer.start()
# for model1, model2 in select_two_models(pop):
#     print(f"Selected models: {pop[model1]} vs {pop[model2]}")
#     child = reproduce(model1, model2, 0.5)
# timer.print_time("Iteratated population in pairs.")

population = survival_of_the_fittest(pop, generate_food(food_size))
population = sort_population(population)

print("Sorted population:")
print_population(population)

population = reproduce_pop(population)
print("Reproduced population:")
print_population(population)
