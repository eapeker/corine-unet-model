import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def build_callbacks(checkpoint_path, lr_config, total_epochs=200, early_stopping_patience=25):
    """Callbacks list"""
    # Create scheduler
    scheduler = create_learning_rate_schedule(
        initial_learning_rate=lr_config['schedule_initial'],
        total_epochs=total_epochs,
        warmup_epochs=lr_config['warmup_epochs'],
        min_lr_ratio=lr_config['min_lr_ratio']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_config['plateau_factor'],
            patience=lr_config['plateau_patience'],
            min_lr=lr_config['min_lr'],
            verbose=1
        ),
        scheduler
    ]
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, callbacks, epochs=200, batch_size=8):
    """Model training"""
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,  # Optimize for memory and performance
        epochs=epochs,    # Total number of epochs
        validation_data=(X_val, y_val),  # Validation set
        callbacks=callbacks,  # Previously defined callbacks
        verbose=1
    )
    return history


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
