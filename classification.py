import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df):
    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))

    X = np.stack(df['embedding'])
    y = df['syndrome_id'].values
    # print(X.shape)
    # print(y.shape)

    return X, y

def apply_knn(X, y, distance_metrics, values, folds):
    results = []
    f1_scores = []
    auc_scores = []
    accuracy_scores = []

    k_values = range(1, values+1)

    for metric in distance_metrics:
        print(f"Evaluation for the distance metric: {metric}")
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

            for train_index, test_index in skf.split(X, y): # split loop
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                knn.fit(X_train, y_train)

                y_pred = knn.predict(X_test)
                y_proba = knn.predict_proba(X_test) # classes prob

                #accuracy
                accuracy = np.mean(y_pred == y_test)
                accuracy_scores.append(accuracy)

                #f1 score
                f1 = f1_score(y_test, y_pred, average='weighted')
                f1_scores.append(f1)

                #auc score
                y_test_onehot = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
                auc = roc_auc_score(y_test_onehot, y_proba, multi_class='ovr', average='macro')
                auc_scores.append(auc)

            mean_accuracy = np.mean(accuracy_scores)
            mean_f1 = np.mean(f1_scores)
            mean_auc = np.mean(auc_scores)
            
            print(f"  k={k}, Accuracy: {mean_accuracy:.3f}, F1-Score: {mean_f1:.3f}, AUC: {mean_auc:.3f}")
            results.append({
                'metric': metric,
                'k': k,
                'accuracy': round(mean_accuracy, 3),
                'f1_score': round(mean_f1, 3),
                'auc': round(mean_auc, 3)
            })
            if k == 15:
                print('\n--------------------------------\n')

    results_df = pd.DataFrame(results)
    results_df.to_csv("knn_results.csv", index=False)


def main(df):
    X, y = prepare_data(df)
    apply_knn(X, y, distance_metrics=['euclidean', 'cosine'], values=15, folds=10)


if __name__ == '__main__':
    df = pd.read_csv("data/output.csv")
    main(df)