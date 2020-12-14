import numpy as np

def estimator(type = "average", samples = [], thresh=None, block_size=None, prev_mean_estimate=None):
    '''
    Returns mean estimator. The three mean estimators used are empirical
    average, truncated empirical average and median of means.

    Input
    -----
    samples: list of samples
    thresh: The truncation parameter for truncated-average. None for others
    block_size: Block size for median of means. None for others
    prev_mean_estimate: Previous mean for fast average calculation

    Returns
    -------
    estimator
    '''
    if type == "average":
        if prev_mean_estimate is not None:
            return (prev_mean_estimate*(len(samples)-1)+samples[-1])/(len(samples)*1.0)
        else:
            arr = np.array(samples)
            return np.mean(arr)
    elif type == "truncated-average":
        samples[samples>thresh] = 0
        samples[samples<-thresh] = 0
        return np.mean(samples)
    elif type == "median-of-means":
        arr = np.array(samples)
        mean_arr = np.mean(arr.reshape(-1, block_size), axis=1)
        return np.median(mean_arr)
    else:
        print("Invalid estimator type.")
        raise ValueError
