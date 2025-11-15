# Evolutionary training loop


import game
from network import model as model_class
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
import hashlib
import io
import sys
import copy

timer = Timer()
fitness_log = []

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
    parameters += f"\t{PERCENT_RANDOM=}\n"
    parameters += f"\t{NUMBER_OF_THREADS=}\n"
    parameters += f"\t{SAVE_DIR=}\n"
    parameters += f"\t{seed=}"

    file_name = os.path.join(SAVE_DIR, "hyperparameters.txt")
    with open(file_name, 'w', ) as file:
        file.write(parameters)
    return

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


def survival_of_the_fittest(population, generation_food=FOOD_SIZE):
    """
    Purpose:
        Feed the old population and updates its fitness.
    Input:
        old_population: A population of models
        generation_food: The number of games to play per generation
    Output:
        a 'new' population, now with updated fitness scores!)
    """
    for i in range(generation_food):
        if generation_food > FOOD_SIZE: print(".", end="")
        sys.stdout.flush() #Makes VScode print
        for model1, model2 in select_two_models(population): 
            game = Mancala()
            game.play_game(model1, model2)
            scores = game.get_score()

            if scores[0] > scores[1]:
                model1.wins += 1
            if scores[1] > scores[0]:
                model2.wins += 1

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
    sorted_population = sorted(population, key=lambda model: model.wins, reverse=True)
    return sorted_population

def mutate(model):
    """
    Purpose:
        Mutates a model by randoming MUTATION_AMOUNT% parameters
    """
    for p in model.parameters():
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

def grow_population(population):
    """
    Idea:
        If it hasn't won, replace it.
        If it won some but not all of its games, mutate it
        If it won all its games, it's allowed to reproduce.
            Afterward, those allowed to reproduce are added to the population.
    """
    repo_pool = []
    replaced = 0
    mutated = 0
    reproduced = 0

    for i, model in enumerate(population):
        if model.wins == 0: #No wins, replace it
            population[i] = model_class()
            replaced += 1
        elif model.wins < FOOD_SIZE: #Some wins, stick around and mutate
            mutate(model)
            mutated += 1
        elif model.wins == FOOD_SIZE: #Add to the reproduction pool!
            repo_pool.append(model)
            reproduced += 1
        else:
            print("model wins outside possible?")
            assert(False)
        model.wins = 0

    for model1, model2 in select_two_models(repo_pool):
        child = reproduce_pair(model1, model2)
        population.append(child)
    print(f"\t{replaced=}\t{mutated=}\t{reproduced=}")
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

    for p1, p2, p_child in zip(model1.parameters(), model2.parameters(), child.parameters()):
        mask = torch.rand_like(p1) < MODEL_SIMILARITY
        p_child.data.copy_(p1 * mask + p2 * (~mask))
    return child

def save_best_models(population, generation):
    """
    Purpose:
        Save the models for later reference or need
    """
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
        To use for later analysis.
    """
    to_log = [model.wins for model in population[:10]]
    to_log.append(len(population))
    fitness_log.append(to_log)

def export_fitness_log(filename=os.path.join("recent_models","fitness_log.csv")):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([1,2,3,4,5,6,7,8,9,10])
        writer.writerows(fitness_log)

def save_population(population, filename=os.path.join("recent_models","current_population.pt")):
    data = []
    for model in population:
        data.append({
            "state_dict": model.state_dict(),
            "fitness": model.fitness,
        })
    torch.save(data, filename)
    print(f"Population with has {population_hash(population)} saved")
    return

def load_population(model_class, filename=os.path.join("recent_models","current_population.pt")):
    data = torch.load(filename)
    population = []
    for entry in data:
        model = model_class()
        model.load_state_dict(entry["state_dict"])
        model.fitness = entry["fitness"]
        population.append(model)
    print(f"Population with has {population_hash(population)} loaded")
    return population

def population_hash(population):
    hasher = hashlib.sha256()
    for model in population:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        hasher.update(buffer.getvalue())

        # Include fitness and any other identifying fields
        hasher.update(str(model.fitness).encode())
    return hasher.hexdigest()


def culling(population, metageneration):
    """
    Purpose:
        This is the hunger games! The population has grown unmanagble and must be culled.
        They will go through many iterations of games.
        If one wins every single game, they are the chosen one.
    """
    rounds = 20*FOOD_SIZE

    print("\n****************************************************************")
    print(f"*********************Begin the {metageneration}st Hunger Games*****************")
    population = population

    for model in population:
        model.wins = 0

    # for m in population:
    #     print(f"{(m.wins,m.fitness)}", end= " ")

    timer.start()
    population = survival_of_the_fittest(population, rounds)
    print()
    timer.print_time("Pop")

    population = sort_population(population)
    save_best_models(population, metageneration)

    #Winner!
    if any([m.wins == rounds for m in population]):
        export_fitness_log()
        save_population(population)
        save_hyperparameters()
        log_top_fitness(population)
        winners = [m for m in population if m.wins == rounds]
        winner = winners[0]
        print(f"A contender ({len(winners)}) has been found!")
        winner.special = True
        for i in range(10):
            print(f"Game {i}:", end="")
            game = Mancala()
            game.move_list = []
            opponent = population[random.randint(1, len(population)-1)]
            game.play_game(winner, opponent)
            print(game.move_list)
        save_best_models(winners, "W")
        return None
    else:
        best = max([m.wins for m in population])
        if best > rounds: 
            for m in population:
                print(f"{(m.wins,m.fitness)}", end= " ")
            print("Error! Too many wins somehow??")
            assert(False)
        survivors = [m for m in population if m.wins == best]
        # for s in survivors: 
        #     s.wins=0
        print(f"The population is not yet ready. Best try was {best}, of which there were {len(survivors)}")
        

        #Make a fresh population and seed it with some survivors
        # population = make_initial_population(POPULATION_SIZE, model_class)
        # population = population + survivors
        
    population = population[:POPULATION_SIZE]
    for model in population:
        model.wins = 0
    return population
    
def evolve(population, generation):
    """
    Purpose:
        Perform a full evolution cycle - one iteration.
    """
    population = survival_of_the_fittest(population, FOOD_SIZE)
    population = sort_population(population)
    log_top_fitness(population)
    grow_population(population)
    return population

def runner():
    try:
        timer.start()
        population = make_initial_population(POPULATION_SIZE, model_class)
        timer.print_time("Initial population.", start="\t")

        metagen = 0
        survivors = []
        while True:
            metagen += 1
            j = 0
            while len(population) < POPULATION_SIZE*10:
                print(f"[{j}] size={len(population)}", end="")
                population = evolve(population, j)
                j += 1
                
            population = culling(population, metagen)
            if population is None:
                break
            
    except Exception as e:
        print("Exception!")
        print(e)
    finally:
 
        print(traceback.format_exc())
        save_population(population)
        export_fitness_log()
        save_hyperparameters()
        print("\nExports complete.")
 