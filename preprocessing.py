import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def create_dataframe(data):
    data_cv = []

    # level 1 - syndrome_id
    for syndrome_id in data.keys():
        # print(f"syndrome_id: {syndrome_id}")

        # level 2 - subject_id
        for subject_id in data[syndrome_id].keys():
            # print(f"    subject_id: {subject_id}")

            # level 3 - image_id
            for image_id in data[syndrome_id][subject_id].keys():
                # print(f"        image_id: {image_id}")
                embedding = data[syndrome_id][subject_id][image_id]
                # print(embedding)
                # print(f"            embedding size: {len(embedding)}")
                # print(type(embedding))
                record = {
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding_size': len(embedding),
                    # 'embedding': embedding
                    'embedding': json.dumps(embedding.tolist())  # save as json string > separate using commas
                }
                data_cv.append(record)

    df = pd.DataFrame(data_cv)
    print('LEN EMBEDDING')
    print(len(df['embedding'].iloc[0]))
    df.to_csv("output.csv", index=False)
    print(df.head())
    return df


def ensure_data(df):
    df['syndrome_id'] = pd.to_numeric(df['syndrome_id'], errors='coerce')
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
    df['image_id'] = pd.to_numeric(df['image_id'], errors='coerce')
    df['embedding_size'] = pd.to_numeric(df['embedding_size'], errors='coerce')
    df['embedding'] = pd.to_numeric(df['embedding'], errors='coerce')

    print(df)
    # print(df.dtypes)
    has_negative = (df  < 0).any().any()

    if has_negative:
        print("Dataframe contains negative values.")
        negative_values = df[df < 0].dropna(how='all')  # filter negative values, dropping nan values
        print("Rows with negative values:")
        print(negative_values)
    else:
        print("No negative values in the dataframe.")

    print(f"\nNull values in each column: \n{df.isnull().sum()}") # ignore embedding nulls

    incorrect_dimensions = df[df['embedding_size'] != 320]
    if incorrect_dimensions.empty:
        print("\nAll embeddings have 320 dimensions.")
    else:
        print("\nEmbeddings with dimensions different from 320:")
        print(incorrect_dimensions)


def get_statistics(df):
    num_syndromes = df['syndrome_id'].nunique() # number of unique syndromes
    print(f"Unique syndromes: {num_syndromes}")

    num_images = df['image_id'].nunique() # number of unique images
    print(f"\nNumber of images: {num_images}")

    images_per_syndrome = df.groupby('syndrome_id')['image_id'].count() # images per syndrome
    print(f"\nImages per syndrome: \n{images_per_syndrome}")
    images_per_syndrome.to_csv('images_per_syndrome.csv', header=['Image Count'])

    top_syndromes = images_per_syndrome.sort_values(ascending=False)
    print(f"\nSyndromes with the most images:\n{top_syndromes.head()}")
    top_syndromes.to_csv('top_syndromes.csv', header=['Image Count'])

    rare_syndromes = images_per_syndrome.sort_values(ascending=True)
    print(f"\nRarest syndromes:\n{rare_syndromes.head()}") # syndromes with the lowest amount of samples
    rare_syndromes.to_csv('rare_syndromes.csv', header=['Image Count'])

    images_per_syndrome.plot(kind='bar', figsize=(12, 6), title='Images per Syndrome')
    plt.xlabel('Syndrome ID')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('images_per_syndrome_plot.png')
    plt.show()
    
    # check if an image is associated with more than one syndrome
    image_syndrome_counts = df.groupby('image_id')['syndrome_id'].nunique()

    # filter images associated with more than one syndrome
    multilabel_images = image_syndrome_counts[image_syndrome_counts > 1]

    if not multilabel_images.empty: # check multilabel
        print("There are images associated with more than one syndrome:")
        print(multilabel_images)
        print(f"\nTotal multilabel images: {len(multilabel_images)}")
    else:
        print("\nNo images are associated with more than one syndrome.")


def main(data):
    df = create_dataframe(data)
    ensure_data(df)
    get_statistics(df)


if __name__ == '__main__':
    pickle_file = 'data/mini_gm_public_v0.1.p'

    with open(pickle_file, "rb") as file:
        data = pickle.load(file)

    main(data) 