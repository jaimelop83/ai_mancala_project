# Evolutionary training loop


import game
from network import model
import torch
import random
from timer import Timer
from network import *
from game import Mancala
from network_tools import random_board
import numpy as np
import cProfile
import pstats
import os
import threading

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
    population = [model_class() for i in range(pop_size)]

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
    for model1, model2 in select_two_models(population):
        #print(f"Selected models: {pop[model1]} vs {pop[model2]}")
        for board in generation_food:
            #scores = [200, 280] #Temp just to be able to run
            #model1.fitness += scores[0]
            #model2.fitness += scores[1]
            game = Mancala(board)
            game.play_game(model1, model2)
            scores = game.get_score()
            model1.fitness += scores[0]
            model2.fitness += scores[1]

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
    #Printed everything
    #for i, model in enumerate(population):
    #    print(f"Model {i}: Fitness: {model.fitness}")

    print("\tPopulation Fitness Report")
    n = len(population)
    percentiles = [100, 99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    for p in percentiles:
        idx =int( (100-p)/100 * (n-1) )
        fitness = population[idx].fitness
        print(f"\t\t{p:>3}%:\t{fitness}")

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
    global MODEL_SIMILARITY

    child = type(model1)()

    for p1, p2, p_child in zip(model1.parameters(), model2.parameters(), child.parameters()):
        mask = torch.rand_like(p1) < MODEL_SIMILARITY
        p_child.data.copy_(p1 * mask + p2 * (~mask))

    if False: #Debugging code to make sure the child is actually a mix of the parents.
        params_child = torch.cat([p.data.view(-1) for p in child.parameters()])
        p1_count = 0
        p2_count = 0
        for p1,p2,c in zip(model1.get_flattened_parameters(), model2.get_flattened_parameters(), params_child):
            if not (p1 == c or p2 == c):
                #print(f"Error: Child parameter {c} does not match either parent {p1} or {p2}.")
                assert(False)
            if p1 ==c:
                p1_count += 1
                #print(".", end="")
            if p2 == c:
                p2_count += 1
                #print("*", end="")
        tot = p1_count + p2_count
        #print(f"{indices=}")
        #print(f"\n{p1_count=} {p2_count=} {tot=}")
        #print(f"{split=}")
        #assert(False)

    return child

#Presently unused (See threading in next function)
def replace_child(population, index, weakest_fit_to_reproduce):
    parent_indices = np.random.randint(weakest_fit_to_reproduce, len(population), size=2)
    new_child_member = reproduce_pair(population[parent_indices[0]], population[parent_indices[1]] )
    population[index] = new_child_member
    return

def reproduce_pop(population):
    """
    Purpose:
        Create a new generation.
    Input:
        population (Note that it MUST be sorted beforehand)
    Output:
        population, with the bottom network.DEATH_RATE members replaced by children from the above network.reproductive_floor.
    """
    global DEATH_RATE, REPRODUCTIVE_FLOOR

    weakest_fit_to_reproduce = int(REPRODUCTIVE_FLOOR * len(population))

    weakest_surviving_model = int(DEATH_RATE * len(population))
    for i in range(-weakest_surviving_model, 0):
        parent_indices = np.random.randint(weakest_fit_to_reproduce, len(population), size=2)
        new_child_member = reproduce_pair(population[parent_indices[0]], population[parent_indices[1]] )
        population[i] = new_child_member

        #An interesting idea ... but much slower?? Might just be threading overhead?
        # threads = [None]*NUMBER_OF_THREADS
        # for j in range(NUMBER_OF_THREADS):
        #     if i + j < 0:
        #         threads[j] = threading.Thread(target=replace_child, args=(population, i + j, weakest_fit_to_reproduce))
        #         threads[j].start()
        # for j in range(NUMBER_OF_THREADS):
        #     if i + j < 0:
        #         threads[j].join()

    return population

def reset_fitness(population):
    """
    Purpose:
        Reset all fitness scores to 0.
    Input:
        population
    Output: 
        population, with all .fitness scores = 0
    """
    for model in population:
        model.fitness = 0
    return population

def save_best_models(population, generation):
    global model
    os.makedirs(SAVE_DIR, exist_ok=True)
    folder = os.path.join(SAVE_DIR, model.model_name)
    os.makedirs(folder, exist_ok=True)

    for i, model in enumerate(population[:10]):
        filename = os.path.join(folder, f"gen_{generation}_model_{i+1}.pt")
        torch.save(model.state_dict(), filename)
    
def evolve(population, generation):
    """
    Purpose:
        Perform a full evolution cycle - one iteration.
    """
    timer.start()
    food = generate_food(FOOD_SIZE)
    timer.print_time("Food", start="\t", end=" ")

    population = survival_of_the_fittest(population, food)
    timer.print_time("Pop", end=" ")

    timer.start()
    population = sort_population(population)
    timer.print_time("Sort", end= " ")

    timer.start()
    save_best_models(population, generation)
    timer.print_time("Saved")

    print_population(population)

    timer.start()
    population = reproduce_pop(population)   
    timer.print_time("Repr", start="\t", end=" ")

    timer.start()
    population = reset_fitness(population)
    timer.print_time("Reset")

    return population

#Proof of concept:
#pop = make_initial_population(POPULATION_SIZE, model)
# timer.start()
# for model1, model2 in select_two_models(pop):
#     print(f"Selected models: {pop[model1]} vs {pop[model2]}")
#     child = reproduce(model1, model2, 0.5)
# timer.print_time("Iteratated population in pairs.")

# population = survival_of_the_fittest(pop, generate_food(FOOD_SIZE))
# population = sort_population(population)

# print("Sorted population:")
# print_population(population)

# population = reproduce_pop(population)
# print("Reproduced population:")
# print_population(population)



def runner():
    timer.start()
    population = make_initial_population(POPULATION_SIZE, model)
    timer.print_time("Initial population created.", start="\t")

    for i in range(NUMBER_OF_GENERATIONS):
        print(f"\nGeneration {i} completed")
        population = evolve(population, i)

        



runner()

# profiler = cProfile.Profile()
# profiler.enable()
# runner()
# profiler.disable()
# #cProfile.run('runner()')

# stats = pstats.Stats(profiler)

# # for func, stat in stats.stats.items():
# #     filename, lineno, funcname = func
# #     shortname = os.path.basename(filename)  # keep only the filename
# #     stats.stats[(shortname, lineno, funcname)] = stats.stats.pop(func)

# stats.sort_stats('tottime')
# stats.dump_stats('profile_data')
# with open("profile_data.txt", "w") as f:
#     stats.stream = f
#     stats.print_stats(50)
#print_population(pop)
    
    
