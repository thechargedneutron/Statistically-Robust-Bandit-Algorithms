from distributions import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def interpret_bandit_instances(filename):
    '''
    Read bandit instances from text file and return bandit instance parameters
    '''
    bandit_instances = {}
    means = []
    with open(filename, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        instance = {}
        splits = data[i].strip().split()
        instance["type"] = splits[0]
        instance["parameters"] = [float(x) for x in splits[1:]]
        instance["samples"] = []
        instance["confidence_bounds"] = []
        instance["mean_estimate"] = 0.0
        bandit_instances[i] = instance
        means.append(distribution_mean(instance["type"], instance["parameters"]))

    bandit_instances["num_arms"] = len(data)
    mean_arr = np.array(means)
    bandit_instances["best_mean"] = np.max(mean_arr)
    return bandit_instances

def plot_regret(cumulative_regrets, labels, title):
    '''
    Deprecated. Please use `plot_with_bars`
    '''
    print("Plotting Cumulative regrets...")
    for i, cumulative_regret in enumerate(cumulative_regrets):
        plt.plot(cumulative_regret, label=labels[i])
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()

def plot_with_bars(cumulative_regrets, labels_list=None, save=False, savepath=None, showbar=True):
    '''
    Plots Cumulative regret with error bars

    Inputs
    ------
    cumulative_regret: a list of 2D matrix with size [num_seeds, num_runs]. Here
                    num_seeds is number of independent simulations of bandits.
    labels_list: list of labels for each element of regret list
    save: boolean. To save or not.
    savepath: Location of path to save
    showbar: boolean. Show error bars or not
    '''
    print("Plotting Cumulative regrets with error bars...")
    fig, ax = plt.subplots(figsize=(5,5))
    colors = ['red', 'g', 'b', 'orange']
    font_size = 16
    for i, cumulative_regret in enumerate(cumulative_regrets):
        avg = np.mean(cumulative_regret, 0)
        std = np.std(cumulative_regret, 0)
        rn = range(1, np.shape(cumulative_regret)[1] + 1)
        if labels_list == None:
            line, = ax.plot(rn, avg, label="{}".format(i), color=colors[i%len(colors)])
        else:
            line, = ax.plot(rn, avg, label=labels_list[i], color=colors[i%len(colors)])
        bar_interval = int(len(cumulative_regret)/10) #To ensure 10 bars
        offset = int(len(cumulative_regret)/30) #Shift bars to avoid obverlap
        if showbar:
            ax.errorbar(rn[offset*i::bar_interval], avg[offset*i::bar_interval], std[offset*i::bar_interval]/2, color=line.get_color(), solid_capstyle='projecting', capsize=5, linestyle='None')
    ax.set_xlabel('Number of rounds (t)', fontsize=font_size)
    ax.set_ylabel('Cumulative regret', fontsize=font_size)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
    ax.legend(loc="upper left", prop={"size":0.8*font_size}, frameon=False)
    ax.figure
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    if save:
        if savepath is None:
            print("savepath must not be None")
            raise ValueError
        plt.savefig(savepath)
    else:
        plt.show()
