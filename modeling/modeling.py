import mlflow
import optuna
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "preprocessing/hotel_bookings_preprocessing.csv"
)

X = df.drop(
    "is_canceled",
    axis=1
)

y = df["is_canceled"]

X_train,X_test,y_train,y_test = \
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MLFLOW (MAIN)
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Hotel Booking Auto Logging"
)

mlflow.autolog()

# ==========================================
# OPTUNA OBJECTIVE  
# ==========================================

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "random_state": 42,

        "eval_metric": "logloss"
    }

    model = XGBClassifier(
        **params
    )

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy"
    ).mean()

    return score

# ==========================================
# OPTUNA
# ==========================================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=10
)

best_params = study.best_params

# ==========================================
# TRAIN FINAL MODEL
# ==========================================

with mlflow.start_run():

    model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = model.score(
        X_test,
        y_test
    )

    print(
        f"Best Parameters: {best_params}"
    )

    for key, value in best_params.items():
        mlflow.log_param(key, value)

    print(
        f"Accuracy: {accuracy}"
    )

    mlflow.log_metric(
        "final_accuracy",
        accuracy
    )