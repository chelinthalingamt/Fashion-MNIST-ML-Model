import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

label_map = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

def exploratory_data_analysis(df):
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values

    def show_image(index):
        image = images[index].reshape(28, 28)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label_map[labels[index]]}")
        plt.axis("off")
        plt.show()

    # Display 5 random images
    for i in np.random.randint(0, len(images), 5):
        show_image(i)

    # Category distribution plot
    plt.figure(figsize=(8,5))
    sns.countplot(x=labels, palette="viridis")
    plt.xticks(ticks=range(10), labels=[label_map[i] for i in range(10)], rotation=45)
    plt.title("Distribution of Clothing Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    from fashion_mnist_level0 import load_dataset
    df = load_dataset()
    exploratory_data_analysis(df)
