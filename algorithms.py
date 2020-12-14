import numpy as np

from functions import *
from distributions import *
from estimators import *

def regret(algo_type="R-UCB", estimator_type="average", bandit_instances=None, horizon=10e3, f="constant", g=None, round_robin_thresh=None, block_size=None, heavy_tail_params=None):
    if algo_type not in ["R-UCB", "R-UCB-G", "R-UCB-G-MOM"]:
        print("Invalid algorithm")
        raise ValueError
    total_regret = 0.0
    for i in range(bandit_instances["num_arms"]):
        bandit_instances[i]["samples"] = [] #Clear sample list everytime
        bandit_instances[i]["best_mean"] = 0.0
    if block_size is not None:
        horizon = horizon // block_size
    else:
        block_size = 1
    if round_robin_thresh is None:
        # We do round robin till number of arms
        round_robin_thresh = bandit_instances["num_arms"] #TODO: Change as per the choice of f
    regrets = np.zeros(horizon)
    print("[INFO] Horizon is {}".format(horizon))
    for t in range(1, int(horizon)+1):
        if t%10000 == 0:
            print("[INFO] Elapsed : {}/{}".format(t, horizon))
        best_arm = get_best_arm(algo_type, bandit_instances, f, g, t, round_robin_thresh, horizon, heavy_tail_params)
        arm = bandit_instances[best_arm]
        for i in range(block_size): #block size is 1 for any-time algorithms
            reward = draw_sample(type=arm["type"], parameters=arm["parameters"])
            if(len(arm["samples"]) == 0):
                arm["samples"] = np.array([reward])
            else:
                arm["samples"] = np.append(arm["samples"], reward)
            regret = bandit_instances["best_mean"] - reward
            total_regret += regret
            regrets[t-1] = total_regret
        if algo_type == "R-UCB-G-MOM":
            arm["mean_estimate"] = estimator(type=estimator_type, samples=arm["samples"], thresh=((heavy_tail_params["B"]*t)/(4*np.log(t)))**(1/(1+heavy_tail_params["epsilon"])), block_size=block_size, prev_mean_estimate=arm["mean_estimate"])
        else:
            arm["mean_estimate"] = estimator(type=estimator_type, samples=arm["samples"], thresh=evaluate_function(t, f["type"], f["base"], f["multiplier"]), block_size=block_size, prev_mean_estimate=arm["mean_estimate"])
    return regrets

def get_best_arm(algo_type="R-UCB", bandit_instances=None, f="constant", g=None, t=0.0, round_robin_thresh=None, horizon=None, heavy_tail_params=None):
    if t <= round_robin_thresh:
        return t%bandit_instances["num_arms"]
    else:
        confidence_bounds = []
        for i in range(bandit_instances["num_arms"]):
            arm = bandit_instances[i]
            if algo_type == "R-UCB":
                width = np.sqrt(evaluate_function(t,  f["type"], f["base"], f["multiplier"])*np.log(t)/len(arm["samples"]))
            elif algo_type == "R-UCB-G":
                width = 1.0/np.log(evaluate_function(t, f["type"], f["base"], f["multiplier"])) + evaluate_function(t, f["type"], f["base"], f["multiplier"])*(16*np.log(t)/len(arm["samples"]))
            elif algo_type == "R-UCB-MOM":
                width = 4.0*(heavy_tail_params["B"]**(1/(1+heavy_tail_params["epsilon"])))*((4*np.log(t)/len(arm["samples"]))**(heavy_tail_params["epsilon"]/(1+heavy_tail_params["epsilon"])))
            else:
                print("Invalid algorithm type")
                raise ValueError
            confidence_bound = arm["mean_estimate"] + width
            confidence_bounds.append(confidence_bound)
            arm["confidence_bounds"].append(confidence_bound)
        return np.argmax(np.array(confidence_bounds))
