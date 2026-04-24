# config.py
# Central configuration for the cardinality estimation project

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "optdb",
    "user": "postgres",
    "password": "1234",
}

# Paths
DATA_DIR = "data/"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5  # for cross-validation

# XGBoost hyperparameters (tune these during experimentation)
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}
