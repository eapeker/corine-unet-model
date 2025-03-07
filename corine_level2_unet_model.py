# Import all libraries at the beginning
import tensorflow as tf
from keras import layers, models
import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from io import BytesIO
from tabulate import tabulate

# Keras imports
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy, MeanIoU

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Other ML libraries
from imblearn.over_sampling import SMOTE


# CORINE Level 2 classes
corine_classes = {
    11: 'Urban fabric',
    12: 'Industrial, comercial and transport units',
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


# NOTE: This is where data would be loaded
# The original code loaded image data (np_img.npy) and labels (np_level_2.npy)
# You should load your data from your local file system
try:
    # Load image data
    images = np.load("path_to_your_data/np_img.npy")
    
    # Load Level 2 labels
    labels2 = np.load("path_to_your_data/np_level_2.npy")

except Exception as e:
    print("Error:", str(e))
    raise


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

# Clean the data
images, labels2 = clean_images_and_labels(images, labels2)


def jaccard_loss(y_true, y_pred, smooth=1e-7):
    """
    Jaccard (IoU) Loss function
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred: Model predictions
        smooth: Small value to prevent division errors
    
    Returns:
        Jaccard Loss value
    """
    # Assume one-hot encoded tensors
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate Jaccard for each class
    class_jaccard_losses = []
    for c in range(y_true.shape[-1]):
        true_class = y_true[..., c]
        pred_class = y_pred[..., c]
        
        intersection = tf.reduce_sum(true_class * pred_class)
        union = tf.reduce_sum(true_class + pred_class) - intersection
        
        jaccard = (intersection + smooth) / (union + smooth)
        class_jaccard_losses.append(1 - jaccard)
    
    # Mean Jaccard Loss
    return tf.reduce_mean(class_jaccard_losses)


# Combined loss function
def combined_loss(y_true, y_pred):
    """Combination of Categorical Cross-Entropy and Jaccard Loss"""
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    jac_loss = jaccard_loss(y_true, y_pred)
    return 0.5 * cce_loss + 0.5 * jac_loss


class custom_MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot encoded tensors to class indices with argmax
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Call the base class MeanIoU method
        return super().update_state(y_true, y_pred, sample_weight)


def create_advanced_unet_model(input_shape=(256, 256, 6), num_classes=15):
    def spatial_attention_block(input_tensor):
        avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)
        concat = layers.Concatenate()([avg_pool, max_pool])
        attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
        return layers.Multiply()([input_tensor, attention])
   
    def aspp_block(inputs, filters=256):
        conv1 = layers.Conv2D(filters, (1, 1), padding='same')(inputs)
        conv2 = layers.Conv2D(filters, (3, 3), dilation_rate=6, padding='same')(inputs)
        conv3 = layers.Conv2D(filters, (3, 3), dilation_rate=12, padding='same')(inputs)
        conv4 = layers.Conv2D(filters, (3, 3), dilation_rate=18, padding='same')(inputs)
        
        global_pool = layers.GlobalAveragePooling2D()(inputs)
        global_pool = layers.Reshape((1, 1, tf.keras.backend.int_shape(inputs)[-1]))(global_pool)
        global_pool = layers.Conv2D(filters, (1, 1), padding='same')(global_pool)
        global_pool = layers.UpSampling2D(size=(tf.keras.backend.int_shape(inputs)[1], tf.keras.backend.int_shape(inputs)[2]), interpolation='bilinear')(global_pool)
        
        concat = layers.Concatenate()([conv1, conv2, conv3, conv4, global_pool])
        return layers.Conv2D(filters, (1, 1), padding='same')(concat)

    def residual_conv_block(input_tensor, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding='same', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(num_filters, (3, 3), padding='same', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        return x

    inputs = layers.Input(input_shape)
    
    c1 = residual_conv_block(inputs, 64)
    c1 = spatial_attention_block(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = residual_conv_block(p1, 128)
    c2 = spatial_attention_block(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = residual_conv_block(p2, 256)
    c3 = spatial_attention_block(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = residual_conv_block(p3, 512)
    c4 = spatial_attention_block(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    bridge = aspp_block(p4)
    
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = layers.concatenate([u6, c4])
    c6 = residual_conv_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = residual_conv_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = residual_conv_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = residual_conv_block(u9, 64)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Create and compile the model
model = create_advanced_unet_model()

optimizer = Adam(learning_rate=0.001)
loss = categorical_crossentropy
metrics = ['accuracy', MeanIoU(num_classes=15)]

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=['accuracy', custom_MeanIoU(num_classes=15)]
)


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

# Normalize images
normalized_images = normalize_data(images)


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

# Main split
X_train, X_test, y_train, y_test = stratified_patch_split(normalized_images, labels2)

# Split again for validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,  # 20% of remaining data for validation
    random_state=42
)


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


def create_learning_rate_schedule(
    initial_learning_rate=5e-4, 
    total_epochs=200,
    warmup_epochs=5,
    min_lr_ratio=0.1
):
    """
    Optimized learning rate scheduler:
    - First 60 epochs: Fast learning in e-4 range
    - 60-100: Transition to e-5 range
    - 100 epoch: Drop to 9e-6
    - 100-200: Gradual decrease from 9e-6 to 1e-6
    """
    def lr_schedule(epoch):
        # First 60 epochs fast learning (e-4 range)
        if epoch < 60:
            if epoch < warmup_epochs:
                return float(initial_learning_rate)
            
            progress = (epoch - warmup_epochs) / (60 - warmup_epochs)
            power = 1.5  # Slightly faster decay
            progress = progress ** (1.0 / power)
            
            cosine_decay = 0.5 * (1 + np.cos(math.pi * progress))
            decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
            return float(initial_learning_rate * decayed)
        
        # 60-100 transition to e-5 range
        elif epoch < 100:
            progress = (epoch - 60) / 40  # 60 to 100
            start_lr = 1e-4
            end_lr = 1.88e-5  # Decrease to 1.88e-5
            return start_lr - ((start_lr - end_lr) * progress)
        
        # At epoch 100 drop to 9e-6 and then gradual decrease
        else:
            if epoch == 100:
                return 9e-6
            
            # 101-200 gradual decrease (9e-6 -> 1e-6)
            remaining_epochs = epoch - 100
            progress = remaining_epochs / 100
            return 9e-6 - (8e-6 * progress)  # Linear decrease from 9e-6 to 1e-6
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)


# Create scheduler
scheduler = create_learning_rate_schedule(
    initial_learning_rate=5e-4,
    total_epochs=200,
    warmup_epochs=5,
    min_lr_ratio=0.1
)

# Callbacks list
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    scheduler
]

# AdamW Optimizer
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=5e-4, 
    weight_decay=1e-5)


# Model training
history = model.fit(
    X_train, y_train,
    batch_size=8,  # Optimize for memory and performance
    epochs=200,    # Total number of epochs
    validation_data=(X_val, y_val),  # Validation set
    callbacks=callbacks,  # Previously defined callbacks
    verbose=1
)


def plot_training_metrics(history):
    # Figure settings
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CORINE Level 2 Model Training Metrics', fontsize=16, y=1.05)
    
    # Loss plot
    axes[0].plot(history.history['loss'], color='blue', linestyle='-', 
                 label='Training Loss', linewidth=1)
    axes[0].plot(history.history['val_loss'], color='orange', linestyle='-', 
                 label='Validation Loss', linewidth=1)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], color='blue', linestyle='-', 
                 label='Training Accuracy', linewidth=1)
    axes[1].plot(history.history['val_accuracy'], color='orange', linestyle='-', 
                 label='Validation Accuracy', linewidth=1)
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # IoU plot with correct metric names
    iou_metric = 'custom__mean_io_u'
    val_iou_metric = 'val_custom__mean_io_u'
    
    axes[2].plot(history.history[iou_metric], color='blue', linestyle='-', 
                 label='Training IoU', linewidth=1)
    axes[2].plot(history.history[val_iou_metric], color='orange', linestyle='-', 
                 label='Validation IoU', linewidth=1)
    axes[2].set_title('Model IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

# Plot training metrics
fig = plot_training_metrics(history)
plt.show()
fig.savefig('training_metrics.png', dpi=300, bbox_inches='tight')


def calculate_class_performance_metrics(true_labels, predicted_labels, corine_classes):
    true_class_indices = np.argmax(true_labels, axis=-1).flatten()
    predicted_class_indices = np.argmax(predicted_labels, axis=-1).flatten()
    
    iou_scores = []
    class_accuracy_scores = []
    class_names = []
    
    for i in range(true_labels.shape[-1]):
        true_mask = (true_class_indices == i)
        predicted_mask = (predicted_class_indices == i)
        
        intersection = np.logical_and(true_mask, predicted_mask).sum()
        union = np.logical_or(true_mask, predicted_mask).sum()
        
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
        
        class_accuracy = accuracy_score(
            true_class_indices[true_mask], 
            predicted_class_indices[true_mask]
        ) if np.sum(true_mask) > 0 else 0
        class_accuracy_scores.append(class_accuracy)
        
        class_names.append(list(corine_classes.values())[i])
    
    performance_df = pd.DataFrame({
        'Class': class_names,
        'IoU': iou_scores,
        'Class Accuracy': class_accuracy_scores
    })
    
    return performance_df


def calculate_test_metrics(y_test, y_pred):
    """
    Calculate class-wise performance metrics for CORINE Level 2 test data
    
    Parameters:
        y_test: Ground truth labels
        y_pred: Model predictions
    """
    # CORINE Level 2 class names
    class_names = list(corine_classes.values())
    
    # Convert from one-hot to class indices
    y_true = np.argmax(y_test, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()
    
    # Calculate metrics
    iou_scores = []
    accuracy_scores = []
    
    for i in range(len(class_names)):
        # Calculate IoU
        true_mask = (y_true == i)
        pred_mask = (y_pred == i)
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
        
        # Calculate Class Accuracy
        true_positive = intersection
        total_true = true_mask.sum()
        accuracy = true_positive / total_true if total_true > 0 else 0
        accuracy_scores.append(accuracy)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'IoU': iou_scores,
        'Class Accuracy': accuracy_scores
    })
    
    return metrics_df

# Evaluate on test set
y_pred_test = model.predict(X_test)
test_metrics = calculate_test_metrics(y_test, y_pred_test)


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (%)", save_path=None):
    """
    Visualize and optionally save the normalized confusion matrix for each class.
    
    Parameters:
    - y_test: Ground truth labels (one-hot encoded)
    - y_pred: Predicted labels (one-hot encoded)
    - title: Plot title
    - save_path: Path to save the figure
    """
    # Convert from one-hot to class indices
    y_true_flat = np.argmax(y_test, axis=-1).flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
    
    # Number of classes
    n_classes = y_test.shape[-1]
    
    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(n_classes))
    
    # Normalize the confusion matrix to percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percentage = np.nan_to_num(cm_percentage)  # Replace NaN with 0
    
    # Class names for CORINE Level 2
    class_names = [
        'Urban', 'Industrial', 'Mine', 'GreenArea', 'Arable',
        'PermanentCrop', 'Pasture', 'MixedAgri', 'Forest', 'Shrub',
        'OpenSpace', 'InlandWet', 'CoastalWet', 'InlandWater', 'Marine'
    ]
    
    # Visualization
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names, 
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save if path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm_percentage

# Generate confusion matrix
conf_matrix_percent = plot_confusion_matrix(y_test, y_pred_test, save_path='confusion_matrix_percentage.png')