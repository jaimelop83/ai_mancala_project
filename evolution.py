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
import csv
import copy
import traceback

timer = Timer()
fitness_log = []

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

#Deprecated - why would we do this?? The initial board state is fixed.
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
        generation_food: The number of games to play per generation
    Output:
        a 'new' population, now with updated fitness scores!)
    """
    for i in range(generation_food):
        print(".", end="")
        for model1, model2 in select_two_models(population): #This could probably be faster with a np.randperm
            game = Mancala()
            game.play_game(model1, model2)
            scores = game.get_score()
            #print(f"\t\tScores added: {scores}")
            model1.fitness += scores[0]
            model2.fitness += scores[1]
    #print()
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

def mutate(model1):
    """
    Purpose:
        Create a mutant from a single model (presumably it is a child just created??)
    """

    for p in model1.parameters():
        if not p.data.numel():
            continue  # skip empty tensors
        num_params = p.data.numel()
        num_to_mutate = int(MUTATION_AMOUNT * num_params)
        if num_to_mutate == 0:
            continue

        # Pick random indices within this tensor
        indices = torch.randperm(num_params)[:num_to_mutate]

        # Mutate those weights in place
        flat = p.data.view(-1)
        flat[indices] = torch.randn(num_to_mutate, device=p.data.device)
    return


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

    if model1.fitness > 0 and model2.fitness > 0:
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

        if random.random() < MUTATION_RATE:
            mutate(child)

        return child
    
    #At least one of the models was cheating.
    else:
        #print("\t\tActually, joking! ", end="")
        if model1.fitness < model2.fitness:
            child = copy.deepcopy(model2)
            #print(f"Used 2nd")
        else:
            child = copy.deepcopy(model1)
            #print(f"Used 1st")
        mutate(child)
        return child

#Presently unused (See threading in next function)
def replace_child(population, index, weakest_fit_to_reproduce):
    parent_indices = np.random.randint(weakest_fit_to_reproduce, len(population), size=2)
    new_child_member = reproduce_pair(population[parent_indices[0]], population[parent_indices[1]] )
    population[index] = new_child_member
    return

#Deprecated in favor or proportoniate selection
def reproduce_pop_old(population):
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
        #print(f"\tReplaced {i=} with a child from {parent_indices[0]}&{parent_indices[1]}")
        new_child_member = reproduce_pair(population[parent_indices[0]], population[parent_indices[1]] )
        population[i] = new_child_member

    return population

def reproduce_pop(population):
    """
    Purpose:
        Create a new generation.
    Input:
        population (Note that it MUST be sorted beforehand)
    Output:
        population with the bottom DEATH_RATE fraction replaced by children
    """
    global DEATH_RATE, REPRODUCTIVE_FLOOR

    #Set up variables
    weakest_fit_to_reproduce = int(REPRODUCTIVE_FLOOR * len(population)) #The number that are fit to reproduce
    repro_pool = population[:weakest_fit_to_reproduce] #Pointers to models that are fit to reproduce
    weakest_surviving_model = int(DEATH_RATE * len(population)) #Where we start replacing

    #Use fitness to make a weight vector for the random selection.
    fitnesses = np.array([max(m.fitness, 0) for m in repro_pool], dtype=float)
    total = fitnesses.sum()
    if total == 0:
        print("Terribly error! All models have negative fitness.")
        assert(False)
    weights = fitnesses / total

    #Reproduce!
    for i in range(-weakest_surviving_model, 0):
        parent_indices = np.random.choice(
            range(weakest_fit_to_reproduce),
            size=2,
            replace=True,
            p=weights
        )

        new_child_member = reproduce_pair(
            population[parent_indices[0]],
            population[parent_indices[1]]
        )
        population[i] = new_child_member

    return population

#Modified to try to keep successful models around... doesn't reset to zero.
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
        if model.fitness > FOOD_SIZE * 100:
            model.fitness = int(model.fitness * FITNESS_DECAY_RATE)
        else:
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

def log_top_fitness(population):
    """
    Purpose:
        Record fitness of top 10 models for this generation.
        """
    top_10 = [model.fitness for model in population[:10]]
    fitness_log.append(top_10)

def export_fitness_log(filename=os.path.join("recent_models","fitness_log.csv")):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([1,2,3,4,5,6,7,8,9,10])
        writer.writerows(fitness_log)

number_of_winners = 0
def check_for_winners(population):
    """
    Purpose:
        If a model wins a game, archive it!
    """
    global number_of_winners
    if number_of_winners > 1000:
        number_of_winners = 0
    os.makedirs(os.path.join(SAVE_DIR, "winners"), exist_ok=True)
    for model in population:
        if model.fitness > 10000:
            filename = os.path.join(SAVE_DIR, "winners", f"winner_{number_of_winners}.pt")
            try:
                torch.save(model.state_dict(), filename)
            except RuntimeError as e:
                print("Unusual error: Torch had trouble saving. Ignoring it.")
                print(e)
                print(traceback.format_exc())
            number_of_winners += 1
    return


def save_population(population, filename=os.path.join("recent_models","current_population.pt")):
    data = []
    for model in population:
        data.append({
            "state_dict": model.state_dict(),
            "fitness": model.fitness
        })
    torch.save(data, filename)

def load_population(model_class, filename=os.path.join("recent_models","current_population.pt")):
    data = torch.load(filename)
    population = []
    for entry in data:
        model = model_class()
        model.load_state_dict(entry["state_dict"])
        model.fitness = entry["fitness"]
        population.append(model)
    return population
    
def evolve(population, generation):
    """
    Purpose:
        Perform a full evolution cycle - one iteration.
    """
    #timer.start()
    #food = generate_food(FOOD_SIZE)
    #timer.print_time("Food", start="\t", end=" ")
    food = FOOD_SIZE

    population = survival_of_the_fittest(population, food)
    timer.print_time("Pop", end=" ")

    timer.start()
    population = sort_population(population)
    timer.print_time("Sort", end= " ")

    timer.start()
    save_best_models(population, generation)
    timer.print_time("Saved", end= " ")

    #print_population(population)
    best = population[0].fitness

    timer.start()
    log_top_fitness(population)
    check_for_winners(population)
    timer.print_time("Archiving", end="")

    timer.start()
    population = reproduce_pop(population)   
    timer.print_time("Repr", end=" ")

    timer.start()
    population = reset_fitness(population)
    timer.print_time("Reset", end="")

    print(f" - {best}")


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


def save_hyperparameters():
    parameters = "Hyperparameters:\n"
    parameters += f"\t{model.model_name=}\n"
    parameters += f"\t{DEATH_RATE=}\n"
    parameters += f"\t{REPRODUCTIVE_FLOOR=}\n"
    parameters += f"\t{MUTATION_RATE=}\n"
    parameters += f"\t{MUTATION_AMOUNT=}\n"
    parameters += f"\t{FITNESS_DECAY_RATE=}\n"
    parameters += f"\t{MODEL_SIMILARITY=}\n"
    parameters += f"\t{FOOD_SIZE=}\n"
    parameters += f"\t{POPULATION_SIZE=}\n"
    parameters += f"\t{NUMBER_OF_GENERATIONS=}\n"
    parameters += f"\t{NUMBER_OF_THREADS=}\n"
    parameters += f"\t{SAVE_DIR=}\n"
    parameters += f"\t{seed=}"

    file_name = os.path.join(SAVE_DIR, "hyperparameters.txt")
    with open(file_name, 'w', ) as file:
        file.write(parameters)
    return



def runner():
    try:
        timer.start()
        if True:
            population = make_initial_population(POPULATION_SIZE, model)
        else:
            population = load_population(model)
            print("Old population loaded")
        timer.print_time("Initial population created.", start="\t")

        for i in range(NUMBER_OF_GENERATIONS):
            population = evolve(population, i)
            #print(f"Generation {i} completed")
            print(f"[{i}]", end="")
    except Exception as e:
        print("Exception!")
        print(e)
        print(traceback.format_exc())
    finally:
        try:
            save_population(population)
            export_fitness_log()
            save_hyperparameters()
            print_population(population)
            print("\nExports complete.")
        except:
            print("\nHold your horses...")
            export_fitness_log()
            save_population(population)
            save_hyperparameters()
            print_population(population)
            print("\nExports complete.")
            





        



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
    
    
