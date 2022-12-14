import numpy as np
import pandas as pd
from scipy import stats
import os
from os.path import exists

dir_path = os.path.dirname(os.path.realpath(__file__))
runs = 10
generations = 10


def t_test(generations, experiment_name_1, experiment_name_2, csv):
    file_path_1 = os.path.join(dir_path, experiment_name_1)
    file_path_2 = os.path.join(dir_path, experiment_name_2)

    df_1 = pd.read_csv(os.path.join(file_path_1, csv))
    df_2 = pd.read_csv(os.path.join(file_path_2, csv))

    mean_per_gen_1 = np.zeros(generations)
    mean_per_gen_2 = np.zeros(generations)
    for gen in range(generations):
        mean_fitness_1 = (df_1['mean'].where(df_1['gen']==gen)).to_numpy()
        mean_fitness_2 = (df_2['mean'].where(df_2['gen']==gen)).to_numpy()
        mean_per_gen_1[gen] = np.mean(mean_fitness_1[~np.isnan(mean_fitness_1)])
        mean_per_gen_2[gen] = np.mean(mean_fitness_2[~np.isnan(mean_fitness_2)])
    t_test = stats.ttest_ind(mean_per_gen_1, mean_per_gen_2)
    return t_test

t_test_enemy_4 = t_test(generations, 'steadystate_4', 'generational_4', 'results.csv')
t_test_enemy_6 = t_test(generations, 'steadystate_6', 'generational_6', 'results.csv')
t_test_enemy_7 = t_test(generations, 'steadystate_7', 'generational_7', 'results.csv')

print("enemy 4 t-test:", t_test_enemy_4)
print("enemy 6 t-test:", t_test_enemy_6)
print("enemy 7 t-test:", t_test_enemy_7)
