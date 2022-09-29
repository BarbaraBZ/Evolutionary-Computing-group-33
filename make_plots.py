import numpy as np
import pandas as pd
import os
from os.path import exists
from matplotlib import pyplot as plt

experiment_name1 = "steadystate_enemy6"
experiment_name2 = ""
file_name_csv = "results.csv"
runs = 10
generations = 10
enemy = 6


def txt_to_csv(experiment_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, experiment_name)
    if not exists(os.path.join(file_path, file_name_csv)):
        print("Writing results to \"" + file_name_csv + "\"")
        results = pd.read_csv(os.path.join(file_path, "results.txt"), error_bad_lines=False)
        results.to_csv(os.path.join(file_path, file_name_csv), index=None)
    else:
        print("Getting results from existing file \"" + file_name_csv + "\"")
        results = pd.read_csv(os.path.join(file_path, file_name_csv))
    return results

def line_plot(runs, generations, experiment_name):
    df = txt_to_csv()
    mean_per_gen = np.zeros(generations)
    var_per_gen = np.zeros(generations)
    max_per_gen = np.zeros(generations)
    var_max_per_gen = np.zeros(generations)
    for gen in range(generations):
        mean_gen = (df['mean'].where(df['gen']==gen)).to_numpy()
        mean_per_gen[gen] = np.mean(mean_gen[~np.isnan(mean_gen)])
        var_gen = (df['std']).where(df['gen']==gen).to_numpy()
        var_per_gen[gen] = np.mean(var_gen[~np.isnan(var_gen)])
        max_gen = (df['best'].where(df['gen']==gen)).to_numpy()
        max_per_gen[gen] = np.mean(max_gen[~np.isnan(max_gen)])
        var_max_per_gen[gen] = np.std(max_gen[~np.isnan(max_gen)])
    return mean_per_gen, max_per_gen, var_per_gen, var_max_per_gen


means_steady, maxs_steady, var_means_steady, var_maxs_steady = line_plot(runs, generations, experiment_name1)
means_generational, maxs_generational, var_means_generational, var_maxs_generational = line_plot(runs, generations, experiment_name2)



x = np.linspace(0, generations, generations)
plt.plot(x, means_steady, label = "mean fitness steadystate")
plt.plot(x, maxs_steady, label = "max fitness steadystate")

plt.plot(x, means_generational, label = "mean fitness generational")
plt.plot(x, maxs_generational, label = "max fitness generational")

plt.fill_between(x, means_steady-var_means_steady, means_steady+var_means_steady, alpha = 0.2)
plt.fill_between(x, maxs_steady-var_maxs, maxs+var_maxs_steady, alpha = 0.2)

plt.fill_between(x, means_generational-var_means_generational, means_generational+var_means_generational, alpha = 0.2)
plt.fill_between(x, maxs_generational-var_maxs_generational, maxs_generational+var_maxs_generational, alpha = 0.2)

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness with std of steadystate and generational against enemy " + str(enemy))
plt.legend()
plt.show()
