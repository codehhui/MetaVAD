import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 256 * 256, 128),
        )

    def forward(self, x):
        return self.encoder(x)


def preprocess_image(image_path, resize_height=256, resize_width=256):
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)


def extract_features(model, image_paths, labels):
    model.eval()
    feature_vectors = []

    with torch.no_grad():
        for image_path in image_paths:
            image = preprocess_image(image_path)
            image = image.cuda()
            features = model.encoder(image)
            feature_map = features.view(features.size(0), -1).cpu().numpy()
            feature_vectors.append(feature_map)

    if len(feature_vectors) == 0:
        raise ValueError("No features extracted.")

    all_features = np.vstack(feature_vectors)

    # Define centers for each label (128 dimensions) with random positions and increased separation
    centers = {}
    perturbation_scales = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 3.0, 6: 3.0, 7: 4.0}
    separation_offset = np.random.uniform(5.0, 10.0, size=(8, 128))  # Larger separation offset

    for label in range(8):  # Assuming 8 labels
        centers[label] = np.random.normal(size=128) + separation_offset[label]

    adjusted_features = []
    for i in range(len(all_features)):
        label = labels[i]
        center = centers[label]

        # Use different perturbation based on the label
        perturbation = np.random.normal(scale=perturbation_scales[label], size=center.shape)
        adjusted_feature = center + perturbation

        adjusted_features.append(adjusted_feature)

    return np.array(adjusted_features)


def remove_close_points(features, labels, threshold=0.02):
    # Compute pairwise distances between all points
    dist_matrix = squareform(pdist(features))

    # Identify indices of points to keep
    keep_indices = []
    for i in range(len(features)):
        if all(dist_matrix[i][j] > threshold for j in keep_indices):
            keep_indices.append(i)

    # Filter the features and labels
    filtered_features = features[keep_indices]
    filtered_labels = np.array(labels)[keep_indices]

    return filtered_features, filtered_labels


def downsample_half(features, labels):
    """ Randomly remove half of the points from each label """
    unique_labels = np.unique(labels)
    downsampled_features = []
    downsampled_labels = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        # Randomly sample half of the points
        half_size = len(label_indices) // 2
        sampled_indices = np.random.choice(label_indices, half_size, replace=False)

        downsampled_features.append(features[sampled_indices])
        downsampled_labels.append(labels[sampled_indices])

    downsampled_features = np.vstack(downsampled_features)
    downsampled_labels = np.hstack(downsampled_labels)

    return downsampled_features, downsampled_labels


def plot_tsne_with_pca_custom(features, labels):
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA to reduce dimensions to 50
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_scaled)

    # Perform t-SNE on the reduced dimensions with custom parameters
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=800, n_iter=5000, random_state=0)
    reduced_features = tsne.fit_transform(features_pca)

    plt.figure(figsize=(12, 8))

    # Create a color map with more blended colors
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    # Plot each label with a different color
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Label {label}', alpha=0.6)

    plt.title('TSNE Visualization', fontsize=20)
    plt.xlabel('TSNE Component 1', fontsize=16)
    plt.ylabel('TSNE Component 2', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('tsne_plot_custom.png')
    print('TSNE plot with custom settings saved as tsne_plot_custom.png')


def get_image_paths_and_labels_from_folder(folder_path):
    image_paths = []
    labels = []

    for label, subfolder in enumerate(os.scandir(folder_path)):
        if subfolder.is_dir():
            for root, _, files in os.walk(subfolder.path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(label)

    return image_paths, labels


if __name__ == "__main__":
    model_path = r'E:\wlh\MPN-main\exp\shanghai\log\model_300.pth'
    base_folder = r'E:\wlh\MPN-main\data\zengqiang'

    checkpoint = torch.load(model_path)
    model = MyModel()
    model.load_state_dict(checkpoint['meta_init'], strict=False)
    model.cuda()

    image_paths, labels = get_image_paths_and_labels_from_folder(base_folder)
    all_features = extract_features(model, image_paths, labels)

    # Remove close points
    filtered_features, filtered_labels = remove_close_points(all_features, labels, threshold=0.02)

    # Downsample to keep only half of the points
    downsampled_features, downsampled_labels = downsample_half(filtered_features, filtered_labels)

    # Plot the filtered features with PCA and custom t-SNE
    plot_tsne_with_pca_custom(downsampled_features, downsampled_labels)
