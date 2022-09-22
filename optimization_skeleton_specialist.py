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

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# headless True for not using visuals (faster), False for visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

hidden = 5      # number of hidden nodes in the neural network
pop_size = 30   # population size
gens = 10       # number of generations
Li = -1         # lower bound for network weights
Ui = 1          # upper bound for network weights
mutation = 0.2  # mutation rate


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
# for running a simulation
def sim(env, x):
    fitness, phealth, ehealth, time = env.play(pcont=x)
    return fitness

    # for normalization (?)

    # enforcing weight limits
def lims(weight):
    if weight >= Ui:
        return Ui
    if weight <= Li:
        return Li

    # variation
    # mutation
def mutate(offspring):
    # kleinere sigma, minder grote verschillen --> uitproberen
    sigma = 2
    # hogere mutation rate, vaker een aanpassing
    mean = 0
    for i in range(0, len(offspring[f])):
        if np.random.unform(0, 1) <= mutation:
            offspring[f][i] = offspring[f][i] + np.random.gaus(mean, sigma)
    return offspring
        # recombination
            # crossover

    # selection (e.g. tournament)
def tournament(pop, k):
    individuals = pop.shape[0]
    winner = randint(0, individuals)
    score = sim(env, winner)
    for i in range(k-1):
        opponent = randint(0, number_individuals)
        opp_score = sim(env, opponent)
        if opp_score > score:
            winner = opponent
            score = opp_score
    return winner

def ranking_selection(pop, fit_pop):
    sorted_fitness = sorted(fit_pop)
    ranks = [sorted_fitness.index(x)+1 for x in fit_pop]
    probs = [rank/len(fit_pop) for rank in ranks]
    parent = np.random.choice(pop, p=probs)
    return parent
        # evaluation
        # doomsday (removing part of the population if nothing improves for a few generations)
            # use a counter for determining whether solu

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

else:
    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

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

for i in range(ini_g+1, gens):
    # create offspring
        # incorporate:
            # parent selection (e.g. tournament)
            # crossover
            # mutation

    # evaluate their fitness

    # perform selection:
        # depends on what type of selection we do, whether we look at entire generation or just children

    # optional: use doomsday to kill the weakest part of the population and replace it with new best/random solutions


    best = np.argmax(fit_pop)   # highest fitness in the new population
    std = np.std(fit_pop)       # std of fitness in the new population
    mean = np.mean(fit_pop)     # mean fitness in the new population

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

env.state_to_log() # checks environment state
