import pandas as pd

def load_dataset(file_path="fashion_mnist.csv"):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
    print("Dataset Shape:", df.shape)
    print(df.head())
    return df

if __name__ == "__main__":
    load_dataset()
