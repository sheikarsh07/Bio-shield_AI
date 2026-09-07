import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── Paths & Setup ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """
    1. Loads raw sensor CSV.
    2. Fills missing values & creates domain features (heat index & dryness).
    3. Balances class distribution via oversampling.
    4. Scales features with StandardScaler & encodes target labels.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file missing at {DATA_PATH}. Run synthetic generator first.")

    df = pd.read_csv(DATA_PATH)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Feature engineering: environmental interaction ratios
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1e-5)
    df["dryness_index"]       = (100 - df["humidity"]) * (100 - df["soil_moisture"])

    X = df.drop(columns=["threat_label"])
    y = df["threat_label"]

    # Target label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Oversample minority classes for balance
    unique_labels, counts = np.unique(y_encoded, return_counts=True)
    max_count = max(counts)
    
    X_balanced, y_balanced = [], []
    for label in unique_labels:
        indices = np.where(y_encoded == label)[0]
        sampled_indices = np.random.choice(indices, size=max_count, replace=True)
        X_balanced.append(X.iloc[sampled_indices])
        y_balanced.extend([label] * max_count)

    X_balanced = pd.concat(X_balanced, ignore_index=True)
    y_balanced = np.array(y_balanced)

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, le


def train_and_evaluate_models():
    """
    Trains Logistic Regression, Random Forest, SVM, and XGBoost.
    Selects and saves the best model based on F1 Score.
    """
    X_train, X_test, y_train, y_test, feature_names, le = load_and_preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "XGBoost":             XGBClassifier(eval_metric="mlogloss", random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")

        # Hyperparameter tuning for Random Forest
        if name == "Random Forest":
            grid = GridSearchCV(
                model,
                {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
                cv=3, scoring="accuracy"
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

        # Evaluation metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, average="weighted")
        rec     = recall_score(y_test, y_pred, average="weighted")
        f1      = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr")

        results[name] = {
            "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1_score": f1, "roc_auc": roc_auc
        }
        print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Rank and select best model
    comparison_df = pd.DataFrame(results).T.drop(columns=["model"])
    best_model_name = comparison_df["f1_score"].idxmax()
    best_model = results[best_model_name]["model"]

    print(f"\nBest Model Selected: {best_model_name}")
    with open(os.path.join(MODELS_DIR, "tabular_threat_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    plot_feature_importance(best_model, best_model_name, feature_names)
    return comparison_df


def plot_feature_importance(model, model_name, feature_names):
    """Generates and saves feature importance chart."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {model_name}")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"))
    plt.close()


if __name__ == "__main__":
    train_and_evaluate_models()

