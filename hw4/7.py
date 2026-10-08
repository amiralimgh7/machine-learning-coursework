import argparse
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASET = "mnist"       # "mnist" / "digits" / "wine"
MAX_SAMPLES = 12000     # فقط برای mnist (برای سرعت). None => کل داده
TEST_SIZE = 0.20
SEED = 42
TARGET_VARIANCE = 0.95

LR_MAX_ITER = 3000
LR_SOLVER = "saga"
LR_NJOBS = -1


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_dataset(name: str, max_samples=None, seed: int = 42):
    name = name.strip().lower()

    if name == "mnist":
        log("Loading dataset: MNIST (OpenML)")
        try:
            ds = fetch_openml("mnist_784", version=1, as_frame=False)
            X = ds.data.astype(np.float32)
            y = ds.target.astype(int)

            if max_samples is not None and max_samples < X.shape[0]:
                log(f"Subsampling MNIST to {max_samples} samples...")
                rng = np.random.default_rng(seed)
                idx = rng.choice(X.shape[0], size=max_samples, replace=False)
                X, y = X[idx], y[idx]

            feature_names = np.array([f"pixel_{i}" for i in range(X.shape[1])])
            dataset_name = f"MNIST (n={X.shape[0]}, d={X.shape[1]})"
            log(f"Loaded: {dataset_name}")
            return X, y, feature_names, dataset_name

        except Exception as e:
            log(f"MNIST fetch failed ({type(e).__name__}). Falling back to DIGITS.")
            name = "digits"

    if name == "digits":
        log("Loading dataset: DIGITS (sklearn)")
        ds = load_digits()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = np.array([f"pixel_{i}" for i in range(X.shape[1])])
        dataset_name = f"Digits (n={X.shape[0]}, d={X.shape[1]})"
        log(f"Loaded: {dataset_name}")
        return X, y, feature_names, dataset_name

    if name == "wine":
        log("Loading dataset: WINE (sklearn)")
        ds = load_wine()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = np.array(ds.feature_names)
        dataset_name = f"Wine (n={X.shape[0]}, d={X.shape[1]})"
        log(f"Loaded: {dataset_name}")
        return X, y, feature_names, dataset_name

    raise ValueError("DATASET باید یکی از این‌ها باشد: 'mnist' / 'digits' / 'wine'")


def fit_evaluate(model, X_train, y_train, X_test, y_test, stage_name: str):
    log(f"Training: {stage_name} ...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    log(f"Training done: {stage_name} | fit_time={fit_time:.4f}s")

    log(f"Predicting: {stage_name} ...")
    t1 = time.perf_counter()
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_time = time.perf_counter() - t1
    log(f"Predicting done: {stage_name} | pred_time={pred_time:.4f}s")

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    return fit_time, pred_time, acc_train, acc_test


def main():
    global DATASET, N_SPLITS
    parser = argparse.ArgumentParser(description="Reproducible coursework experiment")
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.splits < 2: parser.error("--splits must be at least 2")
    DATASET = args.dataset
    N_SPLITS = args.splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.show:
        def save_plot():
            fig = plt.gcf()
            fig.savefig(args.output_dir / f"figure_{fig.number}.png", dpi=160)
            plt.close(fig)
        plt.show = save_plot

    log("Program started.")
    X, y, feature_names, dataset_name = load_dataset(DATASET, MAX_SAMPLES, SEED)

    log("Splitting train/test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    log(f"Split done | Train={X_train.shape} Test={X_test.shape}")

    log("Standardizing + centering (fit on train, transform train/test) ...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    log("Standardization done.")

    clf_base = LogisticRegression(
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=SEED
    )
    base_fit_t, base_pred_t, base_acc_tr, base_acc_te = fit_evaluate(
        clf_base, X_train_s, y_train, X_test_s, y_test, "LogReg WITHOUT PCA"
    )

    log("Fitting full PCA (to compute explained variance curve) on TRAIN ...")
    pca_full = PCA(svd_solver="full", random_state=SEED)
    t0 = time.perf_counter()
    pca_full.fit(X_train_s)
    pca_fit_time = time.perf_counter() - t0
    log(f"Full PCA fit done | time={pca_fit_time:.4f}s")

    evr = pca_full.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    n95 = int(np.searchsorted(cum_evr, TARGET_VARIANCE) + 1)
    log(f"Components needed for >= {int(TARGET_VARIANCE*100)}% variance: n={n95}")

    log("Plotting cumulative explained variance ratio ...")
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cum_evr) + 1), cum_evr, marker="o", markersize=2, linewidth=1)
    plt.axhline(TARGET_VARIANCE, linestyle="--", linewidth=1)
    plt.axvline(n95, linestyle="--", linewidth=1)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA - Cumulative Explained Variance Ratio")
    plt.tight_layout()
    plt.show()
    log("Plot shown.")

    log(f"Fitting PCA with n_components={n95} and transforming data ...")
    pca = PCA(n_components=n95, svd_solver="full", random_state=SEED)
    t1 = time.perf_counter()
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    pca_transform_time = time.perf_counter() - t1
    log(f"PCA fit+transform done | time={pca_transform_time:.4f}s | new_dim={X_train_p.shape[1]}")

    clf_pca = LogisticRegression(
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=SEED
    )
    pca_fit_t, pca_pred_t, pca_acc_tr, pca_acc_te = fit_evaluate(
        clf_pca, X_train_p, y_train, X_test_p, y_test, "LogReg WITH PCA"
    )

    results = pd.DataFrame([
        {
            "Setting": "Without PCA",
            "Features(d)": X_train_s.shape[1],
            "Fit Time (s)": base_fit_t,
            "Predict Time (s)": base_pred_t,
            "Train Acc": base_acc_tr,
            "Test Acc": base_acc_te,
            "Train-Test Gap": base_acc_tr - base_acc_te
        },
        {
            "Setting": f"With PCA ({n95} comps, >=95% var)",
            "Features(d)": X_train_p.shape[1],
            "Fit Time (s)": pca_fit_t,
            "Predict Time (s)": pca_pred_t,
            "Train Acc": pca_acc_tr,
            "Test Acc": pca_acc_te,
            "Train-Test Gap": pca_acc_tr - pca_acc_te
        }
    ])

    log("Final results (Part c):")
    results.to_csv(args.output_dir / "pca_results.csv", index=False)
    print(results.to_string(index=False))

    d_before = X_train_s.shape[1]
    d_after = X_train_p.shape[1]
    total_time_base = base_fit_t
    total_time_pca_pipeline = pca_fit_time + pca_transform_time + pca_fit_t
    gap_base = base_acc_tr - base_acc_te
    gap_pca = pca_acc_tr - pca_acc_te

    log("Summary (Part d):")
    print(f"- Dimensionality: {d_before} -> {d_after}")
    print(f"- Training time: without PCA = {total_time_base:.4f}s | with PCA pipeline = {total_time_pca_pipeline:.4f}s (includes PCA)")
    print(f"- Overfitting proxy (Train-Test gap): without PCA = {gap_base:.4f} | with PCA = {gap_pca:.4f}")

    log("Program finished.")


if __name__ == "__main__":
    main()
