from scipy.stats import randint, uniform

RNDOM_FOREST_PARAMS = {
    'n_estimators':randint(100, 500),
    'max_depth': randint(5,50),
    'max_features' : ['sqrt', 'log2'],
    'criterion' :['squared_error', 'absolute_error']
}

LINEAR_REGRESSION_PARAMS = {

    'fit_intercept': [True, False],
    'positive': [True, False] 
}

RANDOM_SEARCH_PARAMS ={

    'n_iter' : 2,
    'cv' :2,
    'n_jobs' : -1,
    'verbose' : 2,
    'random_state' : 42,
    'scoring' : 'r2' 
}