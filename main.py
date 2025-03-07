from fashion_mnist_loader import load_fashion_mnist
from fashion_mnist_level0 import load_dataset
from fashion_mnist_level1 import exploratory_data_analysis

# Load IDX files and save to CSV (only do this once)
images_path = "images-idx3-ubyte"
labels_path = "labels-idx1-ubyte"
df = load_fashion_mnist(images_path, labels_path)
df.to_csv("fashion_mnist.csv", index=False)

# Load CSV and run EDA
df = load_dataset()
exploratory_data_analysis(df)