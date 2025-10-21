import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

CSV_PATH = "wdbc_data.csv"
TARGET_LABEL = "Diagnosis"
MAP_LABELS = {'M': 1, 'B': 0}
SEED = 42
KFOLDS = 10
SPLITS = [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)]
C_values = 2.0 ** np.arange(-5, 16, 2)
gamma_values = 2.0 ** np.array([-15, -11, -7, -3, 0, 3])
param_grid = {'C': C_values, 'gamma': gamma_values, 'kernel': ['rbf']}


def compute_fscore(X: pd.DataFrame, y: pd.Series):
    scores = {}
    for col in X.columns:
        vals = X[col].astype(float).values
        mean_total = np.mean(vals)
        pos = vals[y == 1]
        neg = vals[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            scores[col] = 0.0
            continue
        mean_pos = np.mean(pos)
        mean_neg = np.mean(neg)
        var_pos = np.var(pos, ddof=0)
        var_neg = np.var(neg, ddof=0)
        numerator = (mean_pos - mean_total) ** 2 + (mean_neg - mean_total) ** 2
        denominator = var_pos + var_neg
        scores[col] = float(numerator / denominator) if denominator != 0 else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def tune_and_evaluate(trainX, trainY, testX, testY):
    svm_model = SVC()
    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    grid = GridSearchCV(svm_model, param_grid, cv=skf, scoring='accuracy', n_jobs=1, verbose=0)
    grid.fit(trainX, trainY)
    best_model = grid.best_estimator_
    predY = best_model.predict(testX)
    accuracy_val = accuracy_score(testY, predY)
    return best_model, grid.best_params_, accuracy_val, predY


def safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        matrix = np.zeros((2, 2), dtype=int)
        for i, actual in enumerate([0, 1]):
            for j, pred in enumerate([0, 1]):
                try:
                    matrix[i, j] = cm[i, j]
                except Exception:
                    matrix[i, j] = 0
        tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fdr = fp / (fp + tp) if (fp + tp) else 0.0
    for_rate = fn / (fn + tn) if (fn + tn) else 0.0
    mcc_val = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return {
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'ppv': ppv,
        'npv': npv,
        'fdr': fdr,
        'forate': for_rate,
        'mcc': mcc_val,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def run_experiment():
    dataset = pd.read_csv(CSV_PATH)
    first_col = dataset.columns[0]
    if first_col.lower() in ('id', 'id_number', 'idnumber', 'index'):
        dataset = dataset.drop(columns=[first_col])
    if 'ID_number' in dataset.columns:
        dataset = dataset.drop(columns=['ID_number'])
    if 'id' in dataset.columns:
        dataset = dataset.drop(columns=['id'])

    if TARGET_LABEL not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_LABEL}' not found in CSV. Columns: {list(dataset.columns)}")
    if dataset[TARGET_LABEL].dtype == object:
        dataset[TARGET_LABEL] = dataset[TARGET_LABEL].map(MAP_LABELS)

    features_all = dataset.drop(columns=[TARGET_LABEL])
    target_all = dataset[TARGET_LABEL].astype(int)

    table2, table3, table4, table6, table7 = {}, {}, {}, {}, {}

    for train_frac, test_frac in SPLITS:
        print(f"\nRunning split {int(train_frac*100)}-{int(test_frac*100)} ...")
        trainX, testX, trainY, testY = train_test_split(
            features_all, target_all, train_size=train_frac, test_size=test_frac,
            stratify=target_all, random_state=SEED
        )

        f_scores = compute_fscore(trainX, trainY)
        table2[f"{int(train_frac*100)}-{int(test_frac*100)}"] = f_scores

        sorted_feats = list(f_scores.index)
        feat_models = []
        acc_list = []

        total_feats = len(sorted_feats)
        for k in range(1, total_feats + 1):
            chosen = sorted_feats[:k]
            feat_models.append(chosen)

            X_train_sel = trainX[chosen]
            X_test_sel = testX[chosen]

            best_model, best_params, accuracy_val, y_pred = tune_and_evaluate(X_train_sel, trainY, X_test_sel, testY)
            acc_list.append(accuracy_val * 100)

            if k == 5:
                metrics = compute_metrics(testY, y_pred)
                table6[f"{int(train_frac*100)}-{int(test_frac*100)}"] = metrics
                table7[f"{int(train_frac*100)}-{int(test_frac*100)}"] = {
                    'confusion': (metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']),
                    'best_params': best_params
                }

            print(f"  Model#{k}: {k} features, test acc = {accuracy_val*100:.4f} %, best_params={best_params}")

        table3[f"{int(train_frac*100)}-{int(test_frac*100)}"] = feat_models
        table4[f"{int(train_frac*100)}-{int(test_frac*100)}"] = acc_list

        print(f"Split {int(train_frac*100)}-{int(test_frac*100)} top features:")
        print(f_scores.head(10))

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    run_experiment()
