import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

def apply_tsne(df):
    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
    embeddings_matrix = np.vstack(df['embedding'])
    tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, random_state=42) # 2d
    embeddings_2d = tsne.fit_transform(embeddings_matrix)

    # add dimensions to df
    df['tsne_x'] = embeddings_2d[:, 0]
    df['tsne_y'] = embeddings_2d[:, 1]
    # df.to_csv('tsne_output_with_coordinates.csv', index=False)
    # unique_syndromes = df['syndrome_id'].nunique()
    # print(unique_syndromes)
    return df

def plot_figure(df):
    plt.figure(figsize=(10, 6))

    for syndrome in df['syndrome_id'].unique():
        subset = df[df['syndrome_id'] == syndrome]
        plt.scatter(subset['tsne_x'], subset['tsne_y'], label=f'Syndrome {syndrome}', alpha=0.7)

    plt.legend(loc='upper right', title='Syndromes', fontsize='xx-small')
    plt.title('t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main(df):
    df_tsne = apply_tsne(df)
    plot_figure(df_tsne)

if __name__ == '__main__':
    df = pd.read_csv('data/output.csv')
    df.to_csv('data/tsne_output.csv', index=False)

    main(df) 