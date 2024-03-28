# ------------------------------------------------------------------------------
# ----------------------------- X G B O O S T ----------------------------------
# ------------------------------------------------------------------------------
#%% Import models's libraries
import xgboost as xgb
from skopt.space import Real, Categorical, Integer

#%% Toggles to go through
rnd_state = 42

#%% Define parameters and model
def model():
    """
    XGBoost for Scikit-learn with default params
    * booster: default= gbtree
        Which booster to use. Can be gbtree, gblinear or dart;
        gbtree and dart use tree based models while gblinear uses linear functions.
    * objective: default=reg:squarederror
        Specify the learning task and the corresponding learning objective.
        The regression objective options are below:
            - reg:squarederror: regression with squared loss.
            - reg:squaredlogerror: regression with squared log loss.
            - reg:logistic: logistic regression.
            - reg:pseudohubererror: regression with Pseudo Huber loss.
            - reg:gamma: gamma regression with log-link.
            - reg:tweedie: Tweedie regression with log-link.
    * n_estimators: int, default=100. The number of trees in the forest.
    * subsample: default=1. Lower ratios avoid over-fitting
    * colsample_bytree: default=1. Lower ratios avoid over-fitting.
    * max_depth: default=6. Lower ratios avoid over-fitting.
    * min_child_weight: default=1. Larger values avoid over-fitting.
    * eta (lr): default=0.3. Lower values avoid over-fitting.
    * reg_lambda: default=1. L2 regularization:. Larger values avoid over-fitting.
    * reg_alpha: default=0. L1 regularization. Larger values avoid over-fitting.
    * gamma: default=0. Larger values avoid over-fitting.
    """
    model_params = {
        # "tree_method": "gpu_hist", #!  for GPU exploiting
        # "gpu_id": 0, #!  for GPU exploiting
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "subsample": 1,
        "colsample_bytree": 1,
        "max_depth": 6,
        "min_child_weight": 1,
        "learning_rate": 0.3,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
        "n_jobs": -1,
        "random_state": rnd_state,
    }
    # fit_params = {
    #     "early_stopping_rounds": 50,
    #     "eval_metric": "mae",
    #     # "eval_metric": "mape",
    #     # "eval_metric": "rmse",
    #     "eval_set": [(X_val, y_val)],
    # }
    model = xgb.XGBRegressor(**model_params)
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
        "subsample": [x / 10 for x in range(5, 11, 1)],
        "colsample_bytree": [x / 10 for x in range(6, 11, 1)],
        "max_depth": [x for x in range(6, 51, 4)],
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "reg_lambda": [1],  # + [x for x in range(1, 11, 1)],
        "reg_alpha": [0],  # + [x for x in range(0, 11, 1)],
        "gamma": [0.0],  # + [x/10 for x in range(5, 60, 5)]
    }
    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    param_grid = {
        "n_estimators": [2400, 2500, 2600],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "max_depth": [6, 12, 24],
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        "learning_rate": [0.01, 0.1, 0.3],
        "reg_lambda": [0.5, 1, 2],
        "reg_alpha": [0, 0.5, 1],
        "gamma": [0, 1, 2, 5],
    }
    # Their core idea of Bayesian Optimization is simple:
    # when a region of the space turns out to be good, it should be explored more.
    # Real: Continuous hyperparameter space.
    # Integer: Discrete hyperparameter space.
    # Categorical: Categorical hyperparameter space.
    bayes_space = {
        "booster": Categorical(["gbtree", "gblinear"]),
        "n_estimators": Integer(100, 3000),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.7, 1.0),
        "max_depth": Integer(3, 10),
        "min_child_weight": Integer(1, 10),
        "learning_rate": Real(0.001, 0.5),
        "reg_lambda": Real(1, 10),
        "reg_alpha": Real(0, 10),
        "gamma": Real(0, 5),
    }

    return param_dist, param_grid, bayes_space
