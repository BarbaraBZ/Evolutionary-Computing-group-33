################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
import random
import time
import concurrent.futures

experiment_name = 'steadystate'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# headless True for not using visuals (faster), False for visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

hidden = 10      # number of hidden nodes in the neural network
pop_size = 100   # population size
gens = 30       # number of generations
Li = -1         # lower bound for network weights
Ui = 1          # upper bound for network weights
mutation = 0.2  # mutation rate
tournament_size = 5
kill_percentage = 0.25

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode='ai',
                  player_controller=player_controller(hidden),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# write environmental variables to the log file
env.state_to_log()

# start timer
ini = time.time()

# training new solutions or testing old ones
run_mode = 'train'

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*hidden + (hidden+1)*5

####### define functions #######

# running a simulation
def sim(x):
    fitness, phealth, ehealth, time = env.play(pcont=x)
    return fitness

# enforcing weight limits
def lims(weight):
    if weight > Ui:
        return Ui
    if weight < Li:
        return Li
    else:
        return weight


# mutate an individual
def mutate(offspring):
    sigma = 0.1
    mean = 0
    mutated = offspring
    for i in range(0, len(offspring[0])):
        x = np.random.uniform(0, 1)
        if x <= mutation:
            mutated[0][i] = offspring[0][i] + np.random.normal(mean, sigma)
    return mutated

# tournament for survivor selection
def tournament(pop, fit_pop, k):
    individuals = pop.shape[0]
    winner = np.random.randint(0, individuals)
    score = fit_pop[winner]
    for i in range(k-1):
        opponent = np.random.randint(0, individuals)
        opp_score = fit_pop[opponent]
        if opp_score > score:
            winner = opponent
            score = opp_score
    return winner

# evaluate fitness
def evaluate(x):
    return np.array(list(map(lambda y: sim(y), x)))

# linear parent selection according to fitness rank
def ranking_selection(pop, fit_pop):
    sorted_fitness = sorted(fit_pop)
    ranks = [sorted_fitness.index(x)+1 for x in fit_pop]
    probs = np.array([rank/sum(ranks) for rank in ranks])
    parent = np.random.choice(len(fit_pop), p=probs)
    return pop[parent]

# doomsday function? (removing part of the population if nothing improves for a few generations)
    # use a counter for determining how long solutions stay the same

# create offspring from parents and mutate them
def crossover(p1, p2):
    # n_offspring = the number of offspring, decided by mutation?
    offspring = np.zeros((1, n_vars))
    # Discrete recombination:
    for j in range(0, len(p1)):
        choice = random.choice([p1[j], p2[j]])
        offspring[0][j] = choice
    offspring = mutate(offspring)
    offspring = np.array(list(map(lambda y: lims(y), offspring[0])))
    return offspring

# original crossover function from the demo
# def crossover(p1, p2):
#     offspring = np.zeros((1, n_vars))
#     cross_prop = np.random.uniform(0,1)
#     offspring[0] = p1*cross_prop+p2*(1-cross_prop)
#     offspring = mutate(offspring)
#     offspring = np.array(list(map(lambda y: lims(y), offspring[0])))
#     return offspring

# doomsday (copied)
# def doomsday(pop,fit_pop):
#
#     worst = int(pop_size/4)  # a quarter of the population
#     order = np.argsort(fit_pop)
#     orderasc = order[0:worst]
#
#     for o in orderasc:
#         for j in range(0,n_vars):
#             pro = np.random.uniform(0,1)
#             if np.random.uniform(0,1)  <= pro:
#                 pop[o][j] = np.random.uniform(Li, Ui) # random dna, uniform dist.
#             else:
#                 pop[o][j] = pop[order[-1:]][0][j] # dna from best
#
#         fit_pop[o]=evaluate([pop[o]])
#
#     return pop,fit_pop

# def doomsday(pop, fit_pop):
#     kill_count = int(pop_size*kill_percentage)
#     order = np.argsort(fit_pop)
#     killed = order[0:kill_count]
#
#     for individual in killed:
#         for gene in range(0, n_vars):
#             prob = np.random.uniform(0,1)
#             if np.random.uniform(0,1) <= prob:
#                 pop[individual][gene] = np.random.uniform(Li, Ui)
#             else:
#                 pop[individual][gene] = pop[order[-1:]][0][gene]
#         fit_pop[individual] = evaluate([pop[individual]])
#     return pop, fit_pop

def doomsday(pop, fit_pop):
    kill_count = int(pop_size*kill_percentage)
    order = np.argsort(fit_pop)
    killed = order[0:kill_count]

    for individual in killed:
        pop[individual] = np.random.uniform(Li, Ui, (1, n_vars))
        fit_pop[individual] = evaluate([pop[individual]])
    return pop, fit_pop

if __name__ == "__main__":

    # loads file with the best solution for testing
    if run_mode =='test':

        bsol = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        evaluate([bsol])

        sys.exit(0)

    # initialize population from old solutions or new ones

    if not os.path.exists(experiment_name+'/evoman_solstate'):

        print( '\nNEW EVOLUTION\n')

        pop = np.random.uniform(Li, Ui, (pop_size, n_vars))
        fit_pop = evaluate(pop)
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        total_best = pop[best]
        total_best_fitness = fit_pop[best]

    else:
        print( '\nCONTINUING EVOLUTION\n')

        env.load_state()
        pop = env.solutions[0]
        fit_pop = env.solutions[1]

        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        total_best = pop[best]
        total_best_fitness = fit_pop[best]

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()


    ####### do the actual evolution here #######

    last_sol = fit_pop[best] # best result of the first generation (or best solution from the previous evolution)
    notimproved = 0 #PART OF DOOMSDAY

    for i in range(ini_g+1, gens):
        # create offspring
        total_offspring = np.zeros((0, n_vars))
        for p in range(0, pop.shape[0], 2):

            # select parents by ranking selection
            # p1 = ranking_selection(pop, fit_pop)
            # p2 = ranking_selection(pop, fit_pop)

            # select parents by tournament selection
            p1 = pop[tournament(pop, fit_pop, 2)]
            p2 = pop[tournament(pop, fit_pop, 2)]

            n_offspring = 2
            for l in range(n_offspring):
                # crossover and mutation
                offspring = crossover(p1, p2)
                total_offspring = np.vstack((total_offspring, offspring))

        # evaluate their fitness (concurrent)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(sim, total_offspring)
            fit_offspring = [result for result in results]
        fit_offspring = np.array(fit_offspring)
        total_pop = np.vstack((pop, total_offspring))
        total_fit = np.append(fit_pop, fit_offspring)
        best = np.argmax(total_fit)
        best_sol = total_fit[best]

        # perform selection (in this case: tournament selection for just the children):
        chosen = [tournament(total_pop, total_fit, tournament_size) for j in range(pop_size)]
        chosen = np.append(chosen[1:], best)
        pop = total_pop[chosen]
        fit_pop = total_fit[chosen]

        # optional: use doomsday to kill the weakest part of the population and replace it with new best/random solutions
        # searching new areas, DOOMSDAY

        notimproved += 1
        if notimproved >= 10:
            file_aux = open(experiment_name + '/results.txt', 'a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop, fit_pop = doomsday(pop, fit_pop)
            notimproved = 0

        best = np.argmax(fit_pop)   # highest fitness in the new population
        std = np.std(fit_pop)       # std of fitness in the new population
        mean = np.mean(fit_pop)     # mean fitness in the new population

        if fit_pop[best] >= total_best_fitness:
            total_best = pop[best]
            total_best_fitness = fit_pop[best]
        # save results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()


    # end timer and print
    fim = time.time()
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    np.savetxt(experiment_name+'/total_best.txt', total_best)

    env.state_to_log() # checks environment state
