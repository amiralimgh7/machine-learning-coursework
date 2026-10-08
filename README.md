# Machine Learning & Computer Vision Coursework

Coursework by Amirali Moghadasi. Completed notebook implementations are grouped by assignment, with two command-line experiments that run on built-in datasets without downloading data.

| Folder | Topics |
|---|---|
| `hw1` | Data cleaning, car-price regression and model comparisons |
| `hw2` | MLE/MAP, logistic regression, text processing and physics-informed neural networks |
| `hw3` | Linear/kernel SVM formulations; neural-network regularization and tuning |
| `hw4` | Bagging/random forests/boosting comparisons, feature importance and PCA |
| `computer-vision` | Custom RGB-to-HSV, histogram equalization, thresholding and bit planes |

## Quick start

Python 3.12, using a virtual environment:

```bash
python -m pip install -r requirements.txt
python hw4/5.py --dataset iris --splits 3 --output-dir outputs/ensemble
python hw4/7.py --dataset digits --output-dir outputs/pca
python -m pytest -q
```

These commands save CSV results and PNG plots without opening windows. Add `--show` for interactive plots. The ensemble script supports `breast_cancer` and `iris`; the PCA script supports `digits`, `wine` and `mnist` and defaults to offline digits. A full MNIST experiment is an explicit option and may need network access.

## Notebooks

```bash
python -m pip install -r requirements-notebooks.txt
python -m pip install jupyterlab
python -m jupyter lab
```

Notebook outputs and execution counts are cleared. HW1 expects its course dataset; HW2 text exercises use NLTK corpora. The SVM notebook's AMPL formulations require `amplpy`, the indicated solvers and your own AMPL license: set `AMPL_LICENSE_UUID` in your environment before its setup cell. No activation UUID is committed. Some SVM functions provide a scikit-learn fallback. HW3 downloads MNIST and its tuner runs can take considerably longer than the smoke tests. The vision notebook provides a deterministic local gradient image if `image_path` is unset; set it to your own file for an image experiment.

## Validation

Five local tests check the logistic gradient against finite differences, finite PINN parameter gradients, primary-color conversion and constant-image equalization, a synthetic SVM decision boundary, and one synthetic training batch for each TensorFlow model. The TensorFlow test skips if that optional package is unavailable; CI installs it. Iris ensemble and digits PCA experiments were run end to end. Full corpus training, AMPL solver sweeps and MNIST hyperparameter searches were not rerun for this release. Dataset scores are coursework results, not application-level performance guarantees.
