import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# CORINE Level 2 classes
corine_classes = {
    11: 'Urban fabric',
    12: 'Industrial, commercial and transport units',
    13: 'Mine, dump and construction sites',
    14: 'Artificial, non-agricultural vegetated areas',
    21: 'Arable land',
    22: 'Permanent crops',
    23: 'Pastures',
    24: 'Heterogeneous agricultural areas',
    31: 'Forest',
    32: 'Shrub and/or herbaceous vegetation associations',
    33: 'Open spaces with little or no vegetation',
    41: 'Inland wetlands',
    42: 'Coastal wetlands',
    51: 'Inland waters',
    52: 'Marine waters'
}


def analyze_band_statistics(band_data, band_number):
    """Analyze band statistics"""
    stats = {
        'min': band_data.min(),
        'max': band_data.max(),
        'mean': band_data.mean(),
        'std': band_data.std(),
        'negative_count': np.sum(band_data < 0),
        'negative_percentage': (np.sum(band_data < 0) / band_data.size) * 100,
        'zero_count': np.sum(band_data == 0),
        'zero_percentage': (np.sum(band_data == 0) / band_data.size) * 100
    }
    return stats


def load_data(images_path, labels_path):
    """Load image data (np_img.npy) and Level 2 labels (np_level_2.npy)"""
    try:
        # Load image data
        images = np.load(images_path)

        # Load Level 2 labels
        labels2 = np.load(labels_path)

    except Exception as e:
        print("Error:", str(e))
        raise

    return images, labels2


# Function to clean images and labels
def clean_images_and_labels(images, labels):
    """
    Cleans images and labels with faulty one-hot encodings

    Args:
        images (np.ndarray): Image array
        labels (np.ndarray): One-hot encoded label array

    Returns:
        tuple: Cleaned images and labels
    """
    # Identify those that don't sum to 1 and those that are zero
    invalid_mask = ~np.isclose(labels.sum(axis=-1), 1.0)

    # Find indices of patches with invalid pixels
    invalid_patch_indices = np.any(invalid_mask, axis=(1,2))

    # Remove invalid patches
    clean_images = images[~invalid_patch_indices]
    clean_labels = labels[~invalid_patch_indices]

    return clean_images, clean_labels


def normalize_data(images):
    """Apply separate normalization for each band"""
    normalized_images = np.zeros_like(images, dtype=np.float32)

    for band in range(images.shape[-1]):
        # Mask negative values
        valid_data = images[:,:,:,band][images[:,:,:,band] >= 0]

        # Calculate 2-98 percentile values
        p2, p98 = np.percentile(valid_data, (2, 98))

        # Apply normalization
        normalized_band = np.clip(images[:,:,:,band], p2, p98)
        normalized_band = (normalized_band - p2) / (p98 - p2)

        # Clip to 0-1 range
        normalized_band = np.clip(normalized_band, 0, 1)

        normalized_images[:,:,:,band] = normalized_band

    return normalized_images


def stratified_patch_split(images, labels, test_size=0.2, random_state=42):
    """Splits patches while preserving class distribution"""
    # Find dominant class in each patch
    dominant_classes = []
    for label in labels:
        # Find the class with highest probability for each pixel
        pixel_classes = np.argmax(label, axis=-1)
        # Get the most frequent class in the patch
        dominant_class = np.bincount(pixel_classes.flatten()).argmax()
        dominant_classes.append(dominant_class)

    # Apply stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_idx, test_idx in sss.split(images, dominant_classes):
        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        return X_train, X_test, y_train, y_test


def split_data(images, labels, seed=42, test_size=0.2, val_size=0.2):
    """Train/test split by dominant class, then a train/validation split"""
    # Main split
    X_train, X_test, y_train, y_test = stratified_patch_split(
        images, labels, test_size=test_size, random_state=seed
    )

    # Split again for validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,  # share of remaining data for validation
        random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def analyze_class_distribution(labels, set_name):
    """Analyze class distribution"""
    # Reduce one-hot encoded labels to a single dimension with argmax
    label_classes = np.argmax(labels, axis=-1)
    unique, counts = np.unique(label_classes, return_counts=True)
    total = np.sum(counts)

    distribution = {}
    class_ids = list(corine_classes.keys())
    for i, (idx, count) in enumerate(zip(unique, counts)):
        class_id = class_ids[i]
        percentage = (count / total) * 100
        distribution[corine_classes[class_id]] = (count, percentage)

    return distribution
