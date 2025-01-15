import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def plot_roc_auc(df):
    metrics = df['metric'].unique()
    plt.figure(figsize=(10, 6))

    for metric in metrics: # get each one
        subset = df[df['metric'] == metric] # each metric = subset
        plt.plot(subset['k'], subset['auc'], label=f"{metric.capitalize()} AUC")
    
    plt.plot([1, 15], [0.5, 1.0], color='gray', linestyle='--', label='AUC = 0.5')
    plt.xlabel("k (Number of neighbors)")
    plt.ylabel("AUC")
    plt.title("ROC AUC Curves for Cosine and Euclidean metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_auc.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_tables(df):
    # table for best results for distance metric
    best_per_metric = []
    for metric in df['metric'].unique():
        subset = df[df['metric'] == metric]
        best_per_metric.append({
            'Distance Metric': metric,
            'Best Accuracy': subset['accuracy'].max(),
            'K (Accuracy)': subset.loc[subset['accuracy'].idxmax(), 'k'],
            'Best F1-Score': subset['f1_score'].max(),
            'K (F1-Score)': subset.loc[subset['f1_score'].idxmax(), 'k'],
            'Best AUC': subset['auc'].max(),
            'K (AUC)': subset.loc[subset['auc'].idxmax(), 'k']
        })
    best_per_metric = pd.DataFrame(best_per_metric)

    # best results overall
    best_overall = pd.DataFrame({
        'Distance Metric': ['Accuracy', 'F1-Score', 'AUC'],
        'Best Value': [
            df['accuracy'].max(),
            df['f1_score'].max(),
            df['auc'].max()
        ],
        'Metric Type': [
            df.loc[df['accuracy'].idxmax(), 'metric'],
            df.loc[df['f1_score'].idxmax(), 'metric'],
            df.loc[df['auc'].idxmax(), 'metric']
        ],
        'K': [
            df.loc[df['accuracy'].idxmax(), 'k'],
            df.loc[df['f1_score'].idxmax(), 'k'],
            df.loc[df['auc'].idxmax(), 'k']
        ]
    })

    print("Best results for each distance metric:")
    print(tabulate(best_per_metric, headers='keys', tablefmt='grid', showindex=False))

    print("\nBest overall results:")
    print(tabulate(best_overall, headers='keys', tablefmt='grid', showindex=False))


    # average results for each metric
    average_metrics = df.groupby('metric').mean(numeric_only=True).reset_index()

    # only metrics columns
    average_metrics = average_metrics[['metric', 'accuracy', 'f1_score', 'auc']]
    average_metrics.rename(columns={
        'metric': 'Distance Metric',
        'accuracy': 'AVG Accuracy',
        'f1_score': 'AVG F1-Score',
        'auc': 'AVG AUC'
    }, inplace=True)

    average_metrics = average_metrics.round(3)

    print("\nAverage Results:")
    print(tabulate(average_metrics, headers='keys', tablefmt='grid', showindex=False))

def main(df):
    plot_roc_auc(df)
    generate_tables(df)

if __name__ == '__main__':
    df = pd.read_csv("data/knn_results.csv")
    main(df)