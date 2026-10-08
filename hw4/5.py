import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier



DATASET = "breast_cancer"
TEST_SIZE = 0.20
RANDOM_STATE_SINGLE = 42

# برای بخش (ب): تعداد تکرار Random Split
N_SPLITS = 30
RANDOM_STATE_REPEATED = 123

# تعداد درخت‌ها (مطابق صورت سوال: Bagging >= 5 و RF >= 10)
BAGGING_TREES = 5
RF_TREES = 10

# تعداد ویژگی‌های نمایش‌داده‌شده در نمودار اهمیت ویژگی
TOP_K_FEATURES = 15



# بارگذاری داده

def load_dataset(name: str):
    name = name.strip().lower()
    if name == "breast_cancer":
        ds = load_breast_cancer()
    elif name == "iris":
        ds = load_iris()
    else:
        raise ValueError("DATASET باید 'breast_cancer' یا 'iris' باشد.")

    X = ds.data
    y = ds.target
    feature_names = getattr(ds, "feature_names", None)
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(X.shape[1])])

    return X, y, np.array(feature_names)



# ساخت مدل‌ها

def build_models(seed: int):
    # (الف-1) Decision Tree با عمق نامحدود
    dt = DecisionTreeClassifier(max_depth=None, random_state=seed)

    # (الف-2) Bagging با حداقل 5 درخت
    base_tree = DecisionTreeClassifier(max_depth=None, random_state=seed)
    bag = BaggingClassifier(
        estimator=base_tree,
        n_estimators=BAGGING_TREES,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )

    # (الف-3) Random Forest با حداقل 10 درخت
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )

    return {
        "DecisionTree": dt,
        f"Bagging_{BAGGING_TREES}": bag,
        f"RandomForest_{RF_TREES}": rf,
    }



# بخش (الف): یک تقسیم تصادفی و گزارش Accuracy

def part_a_single_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE_SINGLE
    )

    models = build_models(RANDOM_STATE_SINGLE)

    rows = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        rows.append({"Model": name, "Accuracy": acc})
        trained[name] = model

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df, (X_train, X_test, y_train, y_test), trained


# بخش (ب): چندین Random Split و محاسبه واریانس/انحراف معیار

def part_b_repeated_splits(X, y):
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_REPEATED
    )

    records = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        # برای اینکه randomness مدل‌ها هم بین تکرارها متفاوت باشد:
        seed = RANDOM_STATE_REPEATED + split_id
        models = build_models(seed)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            records.append({"Split": split_id, "Model": name, "Accuracy": acc})

    df_all = pd.DataFrame(records)

    df_summary = (
        df_all.groupby("Model")["Accuracy"]
        .agg(Mean="mean", Std="std", Var="var", Min="min", Max="max")
        .sort_values("Mean", ascending=False)
        .reset_index()
    )

    return df_all, df_summary


def plot_boxplot_accuracy(df_all):
    models = list(df_all["Model"].unique())
    data = [df_all[df_all["Model"] == m]["Accuracy"].values for m in models]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=models, showmeans=True)
    plt.ylabel("Accuracy")
    plt.title("Distribution of Accuracy over Random Splits (Part b)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


# بخش (ج): Feature Importance برای Random Forest

def part_c_feature_importance(rf_model, feature_names, top_k=15):
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_k = min(top_k, len(feature_names))
    top_idx = order[:top_k]

    df_top = pd.DataFrame({
        "Feature": feature_names[top_idx],
        "Importance": importances[top_idx]
    })

    # چاپ جدول Top Features
    print("\n[Part (c)] Top Feature Importances (Random Forest):")
    print(df_top.to_string(index=False))

    # نمودار
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k)[::-1], importances[top_idx], align="center")
    plt.yticks(range(top_k)[::-1], feature_names[top_idx])
    plt.xlabel("Importance")
    plt.title(f"Random Forest Feature Importance (Top {top_k})")
    plt.tight_layout()
    plt.show()


#
# اجرای کل سوال
def main():
    global DATASET, N_SPLITS
    parser = argparse.ArgumentParser(description="Reproducible coursework experiment")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.splits < 2: parser.error("--splits must be at least 2")
    DATASET = args.dataset
    N_SPLITS = args.splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.show:
        figure_count = 0
        def save_plot():
            nonlocal figure_count
            figure_count += 1
            fig = plt.gcf()
            fig.savefig(args.output_dir / f"figure_{figure_count}.png", dpi=160)
            plt.close(fig)
        plt.show = save_plot

    X, y, feature_names = load_dataset(DATASET)

    # (الف)
    df_a, split_data, trained_models = part_a_single_split(X, y)
    print("[Part (a)] Single Split Accuracy Results:")
    print(df_a.to_string(index=False))

    # (ب)
    df_all, df_b = part_b_repeated_splits(X, y)
    print("\n[Part (b)] Repeated Random Splits Summary (Variance Comparison):")
    print(df_b.to_string(index=False))
    df_a.to_csv(args.output_dir / "single_split.csv", index=False)
    df_b.to_csv(args.output_dir / "repeated_summary.csv", index=False)
    plot_boxplot_accuracy(df_all)

    # (ج) RF را روی همان split بخش (الف) رسم می‌کنیم
    X_train, X_test, y_train, y_test = split_data
    rf_key = f"RandomForest_{RF_TREES}"
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=None,
        random_state=RANDOM_STATE_SINGLE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    part_c_feature_importance(rf, feature_names, top_k=TOP_K_FEATURES)


if __name__ == "__main__":
    main()
