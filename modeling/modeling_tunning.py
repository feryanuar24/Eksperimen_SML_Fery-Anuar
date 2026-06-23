import dagshub
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "preprocessing/hotel_bookings_preprocessing.csv"
)

X = df.drop(
    columns=["is_canceled"]
)

y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MLFLOW (EXPERIMENT)
# ==========================================

dagshub.init(
    repo_owner='feryanuar24',
    repo_name='hotel-booking-mlops',
    mlflow=True
)

mlflow.set_experiment(
    "Hotel Booking Manual Logging"
)

with mlflow.start_run():

    # ======================================
    # PARAMETER
    # ======================================

    n_estimators = 250
    max_depth = 8
    learning_rate = 0.1

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        predictions
    )

    # ======================================
    # LOG PARAMETER
    # ======================================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    # ======================================
    # LOG METRIC
    # ======================================

    mlflow.log_metric(
        "accuracy",
        acc
    )

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,4))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    plt.savefig(
        "modeling/confusion_matrix.png"
    )

    plt.close()

    mlflow.log_artifact(
        "modeling/confusion_matrix.png"
    )

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    report = classification_report(
        y_test,
        predictions
    )

    with open(
        "modeling/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    mlflow.log_artifact(
        "modeling/classification_report.txt"
    )

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    importance = model.feature_importances_

    plt.figure(
        figsize=(12,6)
    )

    plt.bar(
        X.columns,
        importance
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    plt.savefig(
        "modeling/feature_importance.png"
    )

    plt.close()

    mlflow.log_artifact(
        "modeling/feature_importance.png"
    )

    # ======================================
    # MODEL
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        serialization_format="cloudpickle"
    )

    print(
        f"Accuracy: {acc}"
    )