import numpy as np
import pandas as pd
import os
from os.path import exists
from matplotlib import pyplot as plt

experiment_name = "run_test_2"
file_name_csv = "results.csv"
runs = 10
generations = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, experiment_name)


print(file_path)

if not exists(os.path.join(file_path, file_name_csv)):
    print("Writing results to \"" + file_name_csv + "\"")
    results = pd.read_csv(os.path.join(file_path, "results.txt"))
    results.to_csv(os.path.join(file_path, file_name_csv), index=None)
else:
    print("Getting results from existing file \"" + file_name_csv + "\"")
    results = pd.read_csv(os.path.join(file_path, file_name_csv))

def line_plot(runs, generations, df):
    mean_per_gen = np.zeros(generations)
    var_per_gen = np.zeros(generations)
    max_per_gen = np.zeros(generations)
    var_max_per_gen = np.zeros(generations)
    for gen in range(generations):
        mean_gen = (df['mean'].where(df['gen']==gen)).to_numpy()
        mean_per_gen[gen] = np.mean(mean_gen[~np.isnan(mean_gen)])
        var_per_gen[gen] = np.var(mean_gen[~np.isnan(mean_gen)])
        max_gen = (df['best'].where(df['gen']==gen)).to_numpy()
        max_per_gen[gen] = np.mean(max_gen[~np.isnan(max_gen)])
        var_max_per_gen[gen] = np.var(max_gen[~np.isnan(max_gen)])
    return mean_per_gen, max_per_gen, var_per_gen, var_max_per_gen


means, maxs, var_means, var_maxs = line_plot(runs, generations, results)

x = np.linspace(0, generations, generations)
plt.plot(x, means)
plt.plot(x, maxs)
# plt.fill_between(x, means-var_means, means+var_means)
# plt.fill_between(x, maxs-var_maxs, maxs+var_maxs)
plt.show()
