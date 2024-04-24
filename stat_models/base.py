import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import gammaln

# TODO:
### MODEL: add chi squared, out of sample stuff, graphing methods, mean, std, moment generating function

class Model:
    def __init__(self, parameters=None, log_likelihood=None, chi_sq=None, model_type=""):
        self.parameters = parameters if parameters is not None else {}
        self.log_likelihood = log_likelihood
        self.chi_sq = chi_sq
        self.model_type = model_type
        
    def __repr__(self):
        return f"{self.model_type} Model with parameters: {self.parameters}, log likelihood: {self.log_likelihood}"
    
    def __str__(self):
        return f"{self.model_type} Model with parameters: {self.parameters}, log likelihood: {self.log_likelihood}"

def fit_sbg(times: np.ndarray, users: np.ndarray, initial_guess: list = [1, 1]):
    """
    Fits the shifted beta geometric model to the provided times and users data by 
    maximizing the log likelihood. The function returns a Model instance with the fitted
    parameters alpha and beta.

    Parameters:
    times (np.ndarray): A 1D numpy array of times at which observations were made,
                        must be strictly increasing.
    users (np.ndarray): A 1D numpy array of user counts corresponding to each time,
                        must be non-increasing.
    initial_guess (list): A list of two floats representing the initial guesses for alpha and beta.

    Raises:
    ValueError: If inputs are not 1D arrays, not of the same length, don't have at least 2 elements, contain negative values,
                or do not meet the increasing/decreasing requirement.

    Returns:
    Model: An instance of the Model class with fitted parameters, log likelihood,
           chi-squared value (if calculated), and the model type.
    """
    # Validate input arrays
    if not (times.ndim == 1 and users.ndim == 1):
        raise ValueError("Both times and users must be 1D numpy arrays.")
    if len(times) != len(users):
        raise ValueError("times and users arrays must be of the same length.")
    if len(times) < 2:
        raise ValueError("times and users arrays must have at least 2 elements.")
    if not (np.all(times >= 0) and np.all(users >= 0)):
        raise ValueError("Elements of times and users must be non-negative.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times array must be strictly increasing.")
    if not np.all(np.diff(users) <= 0):
        raise ValueError("users array must be non-increasing.")

    # Objective function to minimize (negative log likelihood)
    def objective(params):
        alpha, beta = params
        churned = -np.diff(users)
        p = np.zeros(len(users))
        # first term
        p[0] = alpha / (alpha + beta)

        # middle terms
        for i in range(1, len(p)-1):
            p[i] = p[i-1] * (beta + times[i+1] - 2) / (alpha + beta + times[i+1] - 1)

        # final term
        people_left = users[0] - churned.sum()
        p[len(p)-1] = 1-p.sum()
        churned = np.append(churned, people_left)

        log_likelihood = np.log(p) * churned
        return -log_likelihood.sum()  # Negative because we need to maximize log likelihood

    # Initial guesses for alpha and beta
    initial_guess = [1, 1]

    # Constraints for alpha and beta to be positive
    bounds = [(.001, None), (.001, None)]  # (min, max)

    result = minimize(objective, initial_guess, bounds=bounds)

    # Create and return a Model instance with the fitted parameters
    model = Model(
        parameters={'alpha': result.x[0], 'beta': result.x[1]},
        log_likelihood=-result.fun,
        model_type="Shifted Beta Geometric"
    )
    return model

def fit_nbd(counts: np.ndarray, spikes:list=[], truncated:list=[], shift_start=0, right_censor_from=None)->Model:
    """
    Fits a negative binomial distribution to the provided counts data by maximizing the log likelihood.
    The function returns a Model instance with the fitted parameters r and p.

    Parameters:
    counts (np.ndarray): A 2D numpy array of counts and their associated people, optionally 3D if you want to add time
    spikes (list): A list of values for which there is a spike in the data
    truncated (list): A list values for which there are some "hard core nevers" in the data
    shift_start (int): The index at which to start the shift (default 0)
    right_censor_from (int): The index at which to start right censoring

    Raises:
    ValueError: If inputs are not 1D arrays, not of the same length, don't have at least 2 elements, contain negative values,
                or do not meet the increasing/decreasing requirement.

    Returns:
    Model: An instance of the Model class with fitted
```"""
    # add a column for types
    if counts.shape[0] < 3:
        # Extend the counts array safely
        temp = np.zeros((3, counts.shape[1]))
        temp[:counts.shape[0], :] = counts
        temp[2, :] = 1
        counts = temp

    # Determine the full range of counts from 0 to max in the first row
    max_count = counts[0, :].max()
    full_range = np.arange(0, max_count + 1)

    # Initialize new_counts with 3 rows and full_range columns, third row defaults to 1
    new_counts = np.zeros((3, len(full_range)))
    new_counts[2, :] = 1  # Defaulting third row to 1

    # Populate new_counts with existing data
    for index in range(counts.shape[1]):
        value_index = int(counts[0, index])  # Ensure integer index
        if value_index < len(full_range):
            new_counts[:, value_index] = counts[:, index]

    counts = new_counts

    # Validate input arrays
    if len(counts[0]) < 2:
        raise ValueError("counts must have at least 2 elements")
    if not (np.all(counts[2, :] > 0)):
        raise ValueError("times must be positive")
    if not np.all(counts[0][:shift_start] == 0):
        raise ValueError("shift_start must be less than the first minimum in the counts data")

    def objective(params:list):
        # print(params)
        r, alpha = params[:2]
        spike_parameters = params[2:]
        probabilities = [] # store the probabilities associated with each individual observation of a count
        for idx in range(int(counts[0].max()+1)):
            if idx < shift_start:
                p = 0
                probabilities.append(p)
            elif idx - shift_start == 0: 
                p = (alpha/(alpha + counts[2][idx]))**r
                probabilities.append(p)
            else: 
                p = probabilities[idx-1] * (r+(idx-shift_start)-1)/(idx-shift_start)*(counts[2][idx]/(alpha + counts[2][idx]))
                probabilities.append(p)

        # calculate the probabilities for the spikes and truncated values
        truncated_probabilities = sum([probabilities[trunc] for trunc in truncated])
        spike_probabilities = sum([probabilities[spike] for spike in spikes])
        multiple = 1 - truncated_probabilities # allocate pro rata

        # adjust the probabilties  
        probabilities = [prob / multiple - spike_probabilities for prob in probabilities]
        for idx, spike_prob in enumerate(spikes): 
            probabilities[spike_prob] = spike_parameters[idx] + probabilities[spike_prob]
        # print(counts[1][right_censor_from])

        # right censor the data
        if right_censor_from is not None:
            right_censor_probabilties = sum(probabilities[:right_censor_from])
            # print(1-right_censor_probabilties)
            probabilities[right_censor_from] = 1 - right_censor_probabilties
            counts[1][right_censor_from] = counts[1][right_censor_from:].sum() # allocate all the remaining people in the right spike
            # print(counts[1][right_censor_from])
            counts[1][right_censor_from+1:] = 0 # set counts after that to 0
        
        # print([(p, idx) for p, idx in enumerate(probabilities)])
        # calculate log likelihood
        log_likelihood = counts[1] * np.log(probabilities)
        # print(log_likelihood)
        return -log_likelihood.sum()
    
    # initial guesses for alpha, r, spikes, and anti_spikes
    initial_guess = [1, 1]
    bounds = [(1e-5, None), (1e-5, None)]
    # append the spike probabilities 
    for _ in spikes:
        initial_guess.append(0)
        bounds.append((0, 1))

    result = minimize(objective, initial_guess, bounds=bounds)

    # Create and return a Model instance with the fitted parameters
    parameters={'r': result.x[0], 'alpha': result.x[1]}
    for idx, spike in enumerate(spikes): 
        parameters[f'spike_{spike}'] = result.x[idx+1]

    model = Model(
        parameters=parameters,
        log_likelihood= -result.fun,
        model_type=f"NBD, truncated: {truncated}, spikes: {spikes}, shift_start: {shift_start}, right_censor_from: {right_censor_from}"
    )
    return model