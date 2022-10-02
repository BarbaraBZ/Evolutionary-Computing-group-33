# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
import random
import time
import concurrent.futures

experiment_name = 'generational_4'  #make equal to the experiment you're doing
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# headless True for not using visuals (faster), False for visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

hidden = 10 # number of hidden nodes in the neural network
runs = 10   # number of runs

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[4],      #change to correct enemy
                  playermode='ai',
                  player_controller=player_controller(hidden),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# running a simulation
def sim(x):
    fitness, phealth, ehealth, time = env.play(pcont=x)
    gain = phealth - ehealth
    return gain

# evaluate fitness
def evaluate(x):
    return np.array(list(map(lambda y: sim(y), x)))


if __name__ == "__main__":
    file_aux = open(experiment_name+'/best_results_'+experiment_name+'_test.csv','a')
    file_aux.write('run,mean')
    for j in range(1, runs+1):
        total_fit = []
        for i in range(5):
            # loads file with the best solution for testing
            bsol = np.loadtxt(experiment_name+'/total_best_'+str(j)+'.txt')
            print( '\n RUN'+str(j)+' SIM '+str(i+1)+ '\n')
            total_fit.append(evaluate([bsol]))
        ind_gain = np.mean(total_fit)
        file_aux.write('\n' + str(j) + ',' + str(ind_gain))
    file_aux.close()
