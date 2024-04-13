import numpy as np
from scipy.optimize import minimize, basinhopping

# TODO:
### MODEL: add chi squared, out of sample stuff, graphing methods

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

def fit_sbg(times: np.ndarray, users: np.ndarray, initial_guess: list = [1, 1], holdout: int = None):
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
    holdout (int): The number of observations to hold out from the end of the times and users arrays.

    Raises:
    ValueError: If inputs are not 1D arrays, not of the same length, contain negative values,
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
    if not (np.all(times >= 0) and np.all(users >= 0)):
        raise ValueError("Elements of times and users must be non-negative.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times array must be strictly increasing.")
    if not np.all(np.diff(users) <= 0):
        raise ValueError("users array must be non-increasing.")
    
    if holdout is not None: 
        times = times[:holdout]
        users = users[:holdout]

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
