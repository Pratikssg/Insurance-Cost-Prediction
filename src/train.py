# src/train.py (Simplified for Random Forest Only)
# !/usr/bin/env python3

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["charges"])
    y = df["charges"].copy()
    return X, y


def add_interaction_features(X):
    """Creates new interaction features: smoker * age and smoker * bmi."""
    X_mod = X.copy()
    smoker_binary = X_mod['smoker'].map({'yes': 1, 'no': 0})
    X_mod['smoker_age_interaction'] = smoker_binary * X_mod['age']
    X_mod['smoker_bmi_interaction'] = smoker_binary * X_mod['bmi']
    return X_mod


def build_pipeline():
    """Builds a preprocessing and Random Forest modeling pipeline."""
    numeric_cols = ["age", "bmi", "children", "smoker_age_interaction", "smoker_bmi_interaction"]
    cat_cols = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    return pipeline


def get_param_distributions():
    """Get hyperparameter distributions for Random Forest."""
    return {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [16, 24, 32, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.3, 0.5, 0.7, 1.0],
        "model__bootstrap": [True, False]
    }


def evaluate(y_true, y_pred):
    """Calculates regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main(args):
    mlflow.set_experiment("insurance_cost_experiment")
    X, y = load_data(args.data_path)
    X = add_interaction_features(X)
    y_train_target = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_train_target, test_size=0.2, random_state=RANDOM_STATE
    )

    base_pipeline = build_pipeline()
    registered_model_name = "InsuranceCostModelRF"

    with mlflow.start_run(run_name="RF_Tuned_with_Interactions"):
        print("Running RandomizedSearchCV for Random Forest...")
        param_dist = get_param_distributions()

        search = RandomizedSearchCV(
            base_pipeline,
            param_dist,
            n_iter=args.search_iter,
            cv=4,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        search.fit(X_train, y_train_log)
        selected_model = search.best_estimator_

        print("Best parameters found:", search.best_params_)
        mlflow.log_params(search.best_params_)
        mlflow.set_tag("Tuning_Method", "RF_Interactions_RandomizedSearchCV")

        preds_log = selected_model.predict(X_test)
        preds_original = np.expm1(preds_log)
        y_test_original = np.expm1(y_test_log)

        metrics = evaluate(y_test_original, preds_original)
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_metrics(metrics)
        print("\nFinal Evaluation Metrics (Random Forest):", metrics)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        joblib.dump(selected_model, args.output)
        print(f"Saved optimized RF pipeline locally to: {args.output}")

        mlflow.sklearn.log_model(
            selected_model,
            artifact_path="model_pipeline",
            registered_model_name=registered_model_name
        )
        mlflow.set_tag("R2_score", f"{metrics['r2']:.4f}")

    print(f"Training run finished. Run ID: {mlflow.last_active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/insurance.csv", help="Path to CSV file")
    parser.add_argument("--output", type=str, default="model/pipeline.joblib", help="Output pipeline path")
    parser.add_argument("--search-iter", type=int, default=15, help="Number of RandomizedSearchCV iterations")
    args = parser.parse_args()

    main(args)