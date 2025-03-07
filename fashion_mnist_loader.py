import numpy as np
import struct
import pandas as pd

def load_idx_images(file_path):
    """Reads an IDX file containing images and returns a NumPy array."""
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images

def load_idx_labels(file_path):
    """Reads an IDX file containing labels and returns a NumPy array."""
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_fashion_mnist(images_path, labels_path):
    """Loads Fashion MNIST dataset from IDX files and returns a Pandas DataFrame."""
    images = load_idx_images(images_path)
    labels = load_idx_labels(labels_path)
    
    # Combine labels with images into a single DataFrame
    df = pd.DataFrame(images)
    df.insert(0, "label", labels)  # Insert labels as the first column
    return df

if __name__ == "__main__":
    images_path = "images-idx3-ubyte"
    labels_path = "labels-idx1-ubyte"
    
    df = load_fashion_mnist(images_path, labels_path)
    print("Dataset Loaded Successfully!")
    print("Shape:", df.shape)
    print(df.head())

    # Save as CSV for faster future access
    df.to_csv("fashion_mnist.csv", index=False)
