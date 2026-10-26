import tensorflow as tf


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
