import numpy as np

def evaluate_function(x, type='constant', base=np.e, multiplier=1):
    '''
    Evaluates functions given the type and parameters.
    '''
    if type == 'constant':
        return multiplier
    elif type == 'log':
        return multiplier*np.log(x)/np.log(base)
    elif type == 'shifted-log':
        return multiplier + np.log(x)/np.log(base)
    elif type == "log08":
        return multiplier*(np.log(x)/np.log(base))**0.8
    elif type == "log16":
        return multiplier*(np.log(x)/np.log(base))**1.6
    elif type == 'logsquare':
        return multiplier*(np.log(x)/np.log(base))**2
    elif type == 'log22':
        return multiplier*(np.log(x)/np.log(base))**2.2
    elif type == 'log24':
        return multiplier*(np.log(x)/np.log(base))**2.4
    elif type == 'log25':
        return multiplier*(np.log(x)/np.log(base))**2.5
    elif type == 'shifted-logsquare':
        return multiplier + 0.5*(np.log(x)/np.log(base))**2
    elif type == 'logcube':
        return multiplier*(np.log(x)/np.log(base))**3
    elif type == 'logfour':
        return multiplier*(np.log(x)/np.log(base))**4
    elif type == 'logsix':
        return multiplier*(np.log(x)/np.log(base))**6
    elif type == 'loglog':
        return multiplier*np.log(np.log(x+3)/np.log(base))/np.log(base)
    elif type == 'shifted-loglog':
        return multiplier + np.log(np.log(x+3)/np.log(base))/np.log(base)
    elif type == 'sqrtlog':
        return multiplier*np.sqrt(np.log(x)/np.log(base))
    elif type == 'sqrtloglog':
        return multiplier*np.sqrt(np.log(np.log(x)/np.log(base))/np.log(base))
    elif type == 'logloglog':
        return multiplier*np.log(np.log(np.log(x)/np.log(base))/np.log(base))/np.log(base)
    elif type == "fixed-horizon-f":
        return np.e**((0.5*np.log(np.log(x)))**(1-multiplier))
    elif type == "fixed-horizon-g":
        return 1.0/(np.log(np.log(x)))**multiplier
    else:
        print("Invalid function type")
        raise ValueError
