import numpy as np


def draw_sample(type="uniform", parameters=None):
    '''
    Draws and returns sample from a given list of supported distributions. All
    the distributions can be obtained from samples of uniform distribution. This
    is because we know that the CDF (cumulative distribution function) of any
    random variable is uniformly disttributed in [0, 1]. We use this property to
    obtain samples of all the below mentioned distributions.

    Suppoted distribution and parameters
    ------------------------------------
    - uniform:
        parameters = (a, b)
        returns sample from a uniform distribution in support [a, b]
    - bernoulli:
        parameters = (a, b, p)
        returns sample from a bernoulli distribution with probability p scaled
        from [a, b]
    - exponential:
        parameters = (beta)
        returns sample from exponential distribution with scaling beta
    - pareto:
        parameters = (scale, shape)
        returns sample from pareto distribution with input shape and scaling
    - lomax:
        parameters = (scale, shape)
        returns sample from lomax distribution with input shape and scaling
    '''
    U = np.random.random()
    if type == "uniform":
        low = parameters[0]
        high = parameters[1]
        return U*(high - low) + low
    elif type == "bernoulli":
        low = parameters[0]
        high = parameters[1]
        p = parameters[2]
        if U <= 1 - p:
            return low
        else:
            return high
    elif type == "gaussian":
        mu = parameters[0]
        sigma = parameters[1]
        return np.random.normal(mu, sigma)
    elif type == "exponential":
        beta = parameters[0]
        return (beta) * np.log(1.0/U)
    elif type == "pareto":
        scale = parameters[0]
        shape = parameters[1]
        return scale*((1.0/U)**(1/shape))
    elif type == "lomax":
        scale = parameters[0]
        shape = parameters[1]
        return scale*((1.0/U)**(1.0/shape) - 1)


def distribution_mean(type="uniform", parameters=None):
    '''
    Calculates and returns mean of the distribution.

    Suppoted distribution and parameters
    ------------------------------------
    - uniform:
        parameters = (a, b)
    - bernoulli:
        parameters = (a, b, p)
    - exponential:
        parameters = (beta)
    - pareto:
        parameters = (scale, shape)
    - lomax:
        parameters = (scale, shape)
    '''
    if type == "uniform":
        low = parameters[0]
        high = parameters[1]
        return 0.5*(low + high)
    elif type == "bernoulli":
        low = parameters[0]
        high = parameters[1]
        p = parameters[2]
        return low*(1-p) + high*p
    elif type == "gaussian":
        mu = parameters[0]
        sigma = parameters[0]
        return mu
    elif type == "exponential":
        beta = parameters[0]
        return beta
    elif type == "pareto":
        scale = parameters[0]
        shape = parameters[1]
        if shape <= 1:
            return np.inf
        else:
            return (scale*shape)/(shape - 1)
    elif type == "lomax":
        scale = parameters[0]
        shape = parameters[1]
        if shape <= 1:
            return np.inf
        else:
            print("WIP")
            raise ValueError

if __name__ == "__main__":
    print("Running sanity checks for gaussian distribution...")
    num_iters = 2000
    samples = np.zeros(num_iters)
    for iter in range(num_iters):
        samples[iter] = draw_sample("gaussian", [13.0, 6.0])

    print("Actual mean : {}, Empirical mean : {}".format(13.0, np.mean(samples)))
    print("Actual std : {}, Empirical std : {}".format(6.0, np.std(samples)))
