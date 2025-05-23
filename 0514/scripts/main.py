# -*- coding: utf-8 -*-
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score, f1_score, classification_report # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from xgboost import XGBClassifier # type: ignore
from lightgbm import LGBMClassifier # type: ignore
from catboost import CatBoostClassifier # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import warnings

warnings.filterwarnings("ignore")
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.makedirs("../results", exist_ok=True)

# === 定義模型 ===
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1, force_col_wise=True, n_jobs=1),
    "CatBoost": CatBoostClassifier(save_snapshot=False, logging_level="Silent")
}

# === 參數網格 ===
param_grids = {
    "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ['uniform', 'distance']},
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.1, 0.01]},
    "LightGBM": {"n_estimators": [50, 100], "max_depth": [-1, 5, 10], "learning_rate": [0.1, 0.01]},
    "CatBoost": {"depth": [4, 6], "learning_rate": [0.1, 0.01]}
}

# === 結果儲存 ===
results = []
report_texts = []

def evaluate_and_log(model, name, X_train, X_test, y_train, y_test, param_grid=None):
    if param_grid:
        try:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"✓ {name} 使用最佳參數: {grid.best_params_}")
        except Exception as e:
            print(f"⚠ {name} GridSearch 失敗，使用預設參數: {e}")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
    report = classification_report(y_test, y_pred)
    report_texts.append(f"==== {name} ====" + "\n" + report + "\n")
    print(f"{name} 完成 ✓")
    print(report)
    print("=" * 60)

# === 載入與預處理資料集 ===
datasets = {
    "Wine Quality": pd.read_csv("../data/wine_processed.csv"),
    "Heart Disease": pd.read_csv("../data/heart_processed.csv"),
    "Breast Cancer": pd.read_csv("../data/breast_cancer_processed.csv"),
    "Online Retail": pd.read_csv("../data/online_retail_processed.csv")
}

# === 主迴圈 ===
use_gridsearch = True
use_smote = True

for dataset_name, df in datasets.items():
    
    # 若是 Online Retail，限制最多 3000 筆資料以避免模型過慢
    if dataset_name == "Online Retail":
        df = df.sample(n=3000, random_state=42)
    
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"\n📊 資料集：{dataset_name}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"📌 已對訓練資料使用 SMOTE（{dataset_name}）")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        param_grid = param_grids.get(name) if use_gridsearch else None
        if name in ["SVM", "KNN"]:
            evaluate_and_log(model, f"{dataset_name} - {name}", X_train_scaled, X_test_scaled, y_train, y_test, param_grid)
        else:
            evaluate_and_log(model, f"{dataset_name} - {name}", X_train, X_test, y_train, y_test, param_grid)

# === 匯出結果 ===
results_df = pd.DataFrame(results)
results_df[["Dataset", "Model"]] = results_df["Model"].str.extract(r'(.*) - (.*)')
results_df = results_df.sort_values(by=["Dataset", "F1 Score"], ascending=[True, False])
results_df.to_csv("../results/comparison_results.csv", index=False, encoding="utf-8-sig")

with open("../results/classification_report.txt", "w", encoding="utf-8") as f:
    f.writelines(report_texts)

print("\n📊 模型效能總比較：")
print(results_df.to_string(index=False))
print("\n✅ 所有結果已儲存至 'results/' 資料夾")

from plot_results import plot_comparison_charts
plot_comparison_charts()
