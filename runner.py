import numpy as np

from functions import *
from utils import *
from algorithms import *

#algorithm parameters
horizon = 10000
algorithm = "R-UCB"
est = "average"

# Number of different settings of MAB instances
num_settings = 2
settings = [{"type":"constant", "base":np.e, "multiplier":2.1}, {"type":"logsquare", "base":np.e, "multiplier":1}]
reward_chart = []
num_seeds = 20

#Instance
bandit_instances = interpret_bandit_instances("data2.txt")

for setting in range(num_settings):
    if algorithm == "fixed-horizon":
        batch_size = int(np.ceil(16*np.log(horizon))) #8log(1/delta)
    else:
        batch_size = 1
    cumulative_regret_matrix = np.zeros((num_seeds, (horizon // batch_size)* batch_size))
    if algorithm != "fixed-horizon":
        batch_size = None
    f = settings[setting]
    for seed in range(num_seeds):
        print("[INFO] Starting with random seed {}".format(seed))
        np.random.seed(seed)
        cumulative_regret_matrix[seed][:] = regret(algo_type=algorithm, estimator_type=est, bandit_instances=bandit_instances, horizon=horizon, f=f, block_size=batch_size)
    reward_chart.append(cumulative_regret_matrix)

plot_with_bars(reward_chart)
