import hashlib
import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(LAB_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEMA = {
    "types": {
        "age": "int",
        "workclass": "category",
        "fnlwgt": "int",
        "education": "category",
        "education-num": "int",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": "int",
        "capital-loss": "int",
        "hours-per-week": "int",
        "native-country": "category",
        "income": "category",
    },
    "ranges": {
        "age": (17, 90),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    },
}


def run_ingestion():
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df = dataset.frame.copy()

    if "class" in df.columns and "income" not in df.columns:
        df = df.rename(columns={"class": "income"})

    rows, cols = df.shape
    logging.info("Raw data shape: rows=%d, cols=%d", rows, cols)
    assert rows == 48842 and cols == 15, f"Expected (48842, 15), got ({rows}, {cols})"
    return df


def run_validation(df: pd.DataFrame):
    violations = []

    for col, expected_type in SCHEMA["types"].items():
        if col not in df.columns:
            violations.append(f"Missing column: '{col}'")
            continue
        if expected_type == "int":
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_numeric > 0:
                violations.append(f"Column '{col}' has {non_numeric} non-integer values.")
        elif expected_type == "category":
            if not (isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object):
                violations.append(f"Column '{col}' is not categorical/object.")

    for col, (min_v, max_v) in SCHEMA["ranges"].items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = ((series < min_v) | (series > max_v)).sum()
            if out_of_bounds > 0:
                violations.append(f"Column '{col}' has {out_of_bounds} values outside [{min_v}, {max_v}].")

    for col in df.columns:
        q_count = (df[col].astype(str).str.strip() == "?").sum()
        na_count = df[col].isna().sum()
        if q_count > 0 or na_count > 0:
            logging.info("Column '%s' missing count: '?'=%d, NaN=%d", col, q_count, na_count)

    val_report_path = os.path.join(RESULTS_DIR, "validation_report.txt")
    with open(val_report_path, "w", encoding="utf-8") as f:
        if violations:
            f.write(f"Violations ({len(violations)}):\n")
            for v in violations:
                f.write(f" - {v}\n")
                logging.warning("Violation: %s", v)
        else:
            f.write("Validation passed successfully with 0 violations.\n")
            logging.info("Validation passed with 0 violations.")

    return df


def run_label_quality_audit(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "income"]
    clean_target = df["income"].astype(str).str.strip().str.rstrip(".")

    audit_df = df[feature_cols].copy()
    audit_df["target_clean"] = clean_target

    unique_pairs = audit_df.drop_duplicates()
    conflicting_mask = unique_pairs.duplicated(subset=feature_cols, keep=False)
    conflicting_features_count = unique_pairs[conflicting_mask][feature_cols].drop_duplicates().shape[0]
    logging.info("Duplicate-conflict count: %d", conflicting_features_count)

    sample_50 = df.sample(n=50, random_state=42)
    sample_path = os.path.join(RESULTS_DIR, "audit_sample_50.csv")
    sample_50.to_csv(sample_path, index=False)
    logging.info("Sample 50 saved to: %s", sample_path)


def run_pipeline(df: pd.DataFrame):
    clean_target = df["income"].astype(str).str.strip().str.rstrip(".")
    y = (clean_target == ">50K").astype(int)
    X = df.drop(columns=["income"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    train_combined = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    X_train = train_combined.drop(columns=["income"])
    y_train = train_combined["income"]

    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_medians = X_train[num_cols].median()
    cat_modes = X_train[cat_cols].mode().iloc[0]

    p01 = X_train[num_cols].quantile(0.01)
    p99 = X_train[num_cols].quantile(0.99)

    def transform_split(data: pd.DataFrame, engineer: bool):
        d = data.copy()
        d[num_cols] = d[num_cols].fillna(num_medians)
        d[cat_cols] = d[cat_cols].fillna(cat_modes)
        d[num_cols] = d[num_cols].clip(lower=p01, upper=p99, axis=1)

        if engineer:
            d["capital_net"] = d["capital-gain"] - d["capital-loss"]
            d["hours_per_age"] = d["hours-per-week"] / (d["age"] - 16)
        return d

    metrics = {}
    saved_transformer = {}

    for engineer in [False, True]:
        key = "with_features" if engineer else "without_features"

        X_tr = transform_split(X_train, engineer=engineer)
        X_te = transform_split(X_test, engineer=engineer)

        curr_num = list(num_cols) + (["capital_net", "hours_per_age"] if engineer else [])

        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[curr_num])
        X_te_num = scaler.transform(X_te[curr_num])

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_cat = ohe.fit_transform(X_tr[cat_cols])
        X_te_cat = ohe.transform(X_te[cat_cols])

        X_tr_final = np.hstack([X_tr_num, X_tr_cat])
        X_te_final = np.hstack([X_te_num, X_te_cat])

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr_final, y_train)
        preds = clf.predict(X_te_final)

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds))
        metrics[key] = {"accuracy": acc, "f1": f1}
        logging.info("[%s] Accuracy: %.4f, F1: %.4f", key, acc, f1)

        if engineer:
            saved_transformer = {
                "num_medians": num_medians,
                "cat_modes": cat_modes,
                "p01": p01,
                "p99": p99,
                "scaler": scaler,
                "ohe": ohe,
                "num_cols": curr_num,
                "cat_cols": cat_cols,
            }
            proc_df = pd.DataFrame(X_tr_final)
            proc_df["income"] = y_train.values
            processed_csv_path = os.path.join(RESULTS_DIR, "processed_train.csv")
            proc_df.to_csv(processed_csv_path, index=False)

    preprocessor_path = os.path.join(RESULTS_DIR, "preprocessor.joblib")
    joblib.dump(saved_transformer, preprocessor_path)
    logging.info("Saved preprocessor: %s", preprocessor_path)

    return processed_csv_path, metrics


def finalize(processed_csv_path: str, metrics: dict):
    hasher = hashlib.sha256()
    with open(processed_csv_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    data_hash = hasher.hexdigest()
    logging.info("SHA256: %s", data_hash)

    config_path = os.path.join(RESULTS_DIR, "config.yaml")
    config_body = f"""pipeline:
  name: uci_adult_leakage_free
  random_state: 42
  processed_dataset_sha256: "{data_hash}"
  split:
    train_ratio: 0.70
    test_ratio: 0.30
    stratified: true
  artifacts:
    preprocessor: "results/preprocessor.joblib"
    processed_dataset: "results/processed_train.csv"
  metrics:
    without_features:
      accuracy: {metrics['without_features']['accuracy']:.4f}
      f1: {metrics['without_features']['f1']:.4f}
    with_features:
      accuracy: {metrics['with_features']['accuracy']:.4f}
      f1: {metrics['with_features']['f1']:.4f}
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_body)
    logging.info("Saved config YAML: %s", config_path)


if __name__ == "__main__":
    df_raw = run_ingestion()
    df_val = run_validation(df_raw)
    run_label_quality_audit(df_val)
    proc_csv, exp_metrics = run_pipeline(df_val)
    finalize(proc_csv, exp_metrics)