# ------------------------------------------------------------------------------
# ----------------------- R A N D O M   F O R E S T ----------------------------
# ------------------------------------------------------------------------------
#%% Import models's libraries
from sklearn import ensemble  # RF, Gradient Boosting, AdaBoost
from skopt.space import Real, Categorical, Integer

#%% Toggles to go through
rnd_state = 42

#%% Define parameters and model
def model():
    """A random forest regressor.

    * n_estimators: int, default=100
        The number of trees in the forest.
    * criterion: {"mse", "mae"}, default="mse"
        The function to measure the quality of a split.
    * max_depth: int, default=None
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or until
        all leaves contain less than min_samples_split samples.
    * min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node:
    * min_samples_leaf: int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least
        min_samples_leaf training samples in each of the left and right branches.
    * min_weight_fraction_leaf: float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
        Samples have equal weight when sample_weight is not provided.
    * max_features: {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split.
    * max_samples: int or float, default=None
        If bootstrap is True, the number of samples to draw from X to train each
        base tree.
    * bootstrap: bool, default=True
        Whether bootstrap samples are used when building trees.
    * oob_scorebool, default=False
        Whether to use out-of-bag samples to estimate the R^2 on unseen data
    """
    model_params = {
        "n_estimators": 100,
        "criterion": "mse",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "auto",
        "max_samples": None,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,
        "random_state": rnd_state,
    }
    model = ensemble.RandomForestRegressor(**model_params)
    model_name = type(model).__name__

    return (model, model_name, model_params)


#%% --------------------------- GridSearchCV -----------------------------------
def param_search():
    # Parameters' distributions tune in case RANDOMIZED grid search
    ## Dictionary with parameters names (str) as keys and distributions
    ## or lists of parameters to try.
    ## If a list is given, it is sampled uniformly.
    param_dist = {
        "n_estimators": [x for x in range(50, 1001, 50)],
        "criterion": ["mse", "mae"],
        "max_depth": [None] + [x for x in range(6, 51, 4)],
        "min_samples_split": [x for x in range(2, 7, 2)],
        "min_samples_leaf": [x for x in range(1, 5, 1)],
        "max_features": ["auto", "sqrt", "log2"],
        "max_samples": [x / 10 for x in range(5, 11, 1)],
    }
    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    param_grid = {
        "n_estimators": [500, 750, 1000],
        "max_depth": [None] + [6, 12, 24],
        "min_samples_split": [x for x in range(2, 7, 2)],
        "min_samples_leaf": [x for x in range(1, 5, 1)],
    }
    # Their core idea of Bayesian Optimization is simple:
    # when a region of the space turns out to be good, it should be explored more.
    # Real: Continuous hyperparameter space.
    # Integer: Discrete hyperparameter space.
    # Categorical: Categorical hyperparameter space.
    bayes_space = {
        "n_estimators": Integer(100, 1000),
        "criterion": Categorical(["mse", "mae"]),
        "max_depth": Integer(3, 500),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(2, 10),
        "max_features": Categorical(["auto", "sqrt", "log2"]),
    }

    return param_dist, param_grid, bayes_space
