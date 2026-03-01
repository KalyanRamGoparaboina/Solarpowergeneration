import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

TARGET_COL = "power-generated"

print("=" * 80)
print("SOLAR POWER GENERATION - MODEL TRAINING AND DEPLOYMENT")
print("=" * 80)

# Check for CSV file (accept command-line argument or default)
csv_file = sys.argv[1] if len(sys.argv) > 1 else "solarpowergeneration-1.csv"

print("\n[1/6] Loading Dataset...")
print(f"Looking for: {csv_file}")

if not os.path.exists(csv_file):
    print(f"\nERROR: CSV file not found: '{csv_file}'")
    print(f"\nCurrent directory: {os.getcwd()}")
    print("\nCSV files in current directory:")
    for file in os.listdir("."):
        if file.endswith(".csv"):
            print(f"  - {file}")

    print("\nSOLUTIONS:")
    print("  Option 1: Copy CSV to current directory")
    print("  Option 2: Run with path: python Solarpowergenerationdeployment.py <path_to_csv>")
    print("  Option 3: Use Streamlit dashboard (streamlit run app.py) and upload CSV there")
    print("\n" + "=" * 80)
    sys.exit(1)

df = pd.read_csv(csv_file)
df.columns = [col.strip() for col in df.columns]

if TARGET_COL not in df.columns:
    print(f"ERROR: Missing required target column '{TARGET_COL}'.")
    sys.exit(1)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL not in numeric_cols:
    print(f"ERROR: Target column '{TARGET_COL}' must be numeric.")
    sys.exit(1)

feature_cols = [col for col in numeric_cols if col != TARGET_COL]
if not feature_cols:
    print("ERROR: No numeric feature columns found.")
    sys.exit(1)

print("OK: Dataset loaded successfully")
print(f"  Rows: {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"  Numeric features used: {len(feature_cols)}")
print(f"  Target range: {df[TARGET_COL].min()} - {df[TARGET_COL].max()}")

print("\n[2/6] Data Preprocessing...")
missing_before = df[numeric_cols].isnull().sum().sum()
df_clean = df[numeric_cols].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
print(f"OK: Missing values handled: {missing_before} value(s) imputed with median")

X = df_clean[feature_cols]
y = df_clean[TARGET_COL]

print(f"OK: Features: {len(feature_cols)}")
print(f"OK: Target: {TARGET_COL}")

if len(df_clean) < 10:
    print("ERROR: Need at least 10 rows to train and evaluate models.")
    sys.exit(1)

print("\n[3/6] Train-Test Split (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"OK: Training: {X_train.shape[0]} samples")
print(f"OK: Testing: {X_test.shape[0]} samples")

print("\n[4/6] Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("OK: StandardScaler applied")

print("\n[5/6] Training Models...")
print("-" * 80)

models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "is_scaled": True,
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42, max_depth=15),
        "X_train": X_train,
        "X_test": X_test,
        "is_scaled": False,
    },
    "Random Forest": {
        "model": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
        "X_train": X_train,
        "X_test": X_test,
        "is_scaled": False,
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        "X_train": X_train,
        "X_test": X_test,
        "is_scaled": False,
    },
    "SVR": {
        "model": SVR(kernel="rbf", C=100, gamma="auto"),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "is_scaled": True,
    },
}

results = []
best_model = None
best_r2 = -np.inf
best_model_name = None
best_is_scaled = False
best_rmse = None
best_mae = None

for name, config in models.items():
    print(f"\nTraining {name}...")

    model = config["model"]
    X_tr = config["X_train"]
    X_te = config["X_test"]

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    cv_folds = min(5, len(y_train))
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean = np.nan
        cv_std = np.nan

    results.append(
        {
            "Model": name,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "CV Mean": round(cv_mean, 4) if not np.isnan(cv_mean) else np.nan,
            "CV Std": round(cv_std, 4) if not np.isnan(cv_std) else np.nan,
        }
    )

    print(f"  R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    if not np.isnan(cv_mean):
        print(f"  {cv_folds}-Fold CV: {cv_mean:.4f} (+/- {cv_std:.4f})")
    else:
        print("  CV skipped: not enough rows")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name
        best_is_scaled = config["is_scaled"]
        best_rmse = rmse
        best_mae = mae

print("\n" + "=" * 80)
print("MODEL COMPARISON RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n" + results_df.to_string(index=False))

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name}")
print("=" * 80)
print(f"  R2 Score: {best_r2:.4f}")
print(f"  RMSE: {best_rmse:.2f}")
print(f"  MAE: {best_mae:.2f}")

print("\n[6/6] Saving Model & Visualizations...")

model_data = {
    "model": best_model,
    "scaler": scaler if best_is_scaled else None,
    "is_scaled": best_is_scaled,
    "best_model_name": best_model_name,
    "best_r2": best_r2,
    "best_rmse": best_rmse,
    "best_mae": best_mae,
    "feature_columns": feature_cols,
    "results": results_df,
}

with open("solar_power_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("OK: Model saved: solar_power_model.pkl")

plt.style.use("seaborn-v0_8-darkgrid")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].bar(
    results_df["Model"],
    results_df["R2"],
    color=["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"],
    alpha=0.8,
)
axes[0, 0].set_ylabel("R2 Score", fontsize=12)
axes[0, 0].set_xlabel("Model", fontsize=12)
axes[0, 0].set_title("Model R2 Comparison", fontsize=14, fontweight="bold")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(alpha=0.3)

x = np.arange(len(results_df))
width = 0.35
axes[0, 1].bar(x - width / 2, results_df["RMSE"], width, label="RMSE", color="#ef4444", alpha=0.7)
axes[0, 1].bar(x + width / 2, results_df["MAE"], width, label="MAE", color="#f59e0b", alpha=0.7)
axes[0, 1].set_ylabel("Error Value", fontsize=12)
axes[0, 1].set_title("RMSE and MAE Comparison", fontsize=14, fontweight="bold")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(results_df["Model"], rotation=45, ha="right")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

if best_is_scaled:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

axes[1, 0].scatter(y_test, y_pred_best, alpha=0.5, s=30, color="#3b82f6")
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[1, 0].set_xlabel("Actual Power (Joules)", fontsize=12)
axes[1, 0].set_ylabel("Predicted Power (Joules)", fontsize=12)
axes[1, 0].set_title(f"{best_model_name}: Actual vs Predicted", fontsize=14, fontweight="bold")
axes[1, 0].grid(alpha=0.3)

residuals = y_test - y_pred_best
axes[1, 1].scatter(y_pred_best, residuals, alpha=0.5, s=30, color="#10b981")
axes[1, 1].axhline(y=0, color="r", linestyle="--", lw=2)
axes[1, 1].set_xlabel("Predicted Power", fontsize=12)
axes[1, 1].set_ylabel("Residuals", fontsize=12)
axes[1, 1].set_title("Residual Plot", fontsize=14, fontweight="bold")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=300, bbox_inches="tight")
print("OK: Chart saved: model_evaluation.png")

fig2, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df_clean.corr(numeric_only=True)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
print("OK: Chart saved: correlation_heatmap.png")

if best_model_name in ["Random Forest", "Gradient Boosting"]:
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    print("\n" + feature_importance_df.to_string(index=False))

    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="#3b82f6", alpha=0.7)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance - {best_model_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    print("\nOK: Chart saved: feature_importance.png")

print("\n" + "=" * 80)
print("OK: DEPLOYMENT COMPLETE")
print("=" * 80)
print("\nGenerated Files:")
print("  - solar_power_model.pkl")
print("  - model_evaluation.png")
print("  - correlation_heatmap.png")
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    print("  - feature_importance.png")
print("\nNext: streamlit run app.py")
print("=" * 80)
