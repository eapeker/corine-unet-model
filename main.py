import argparse
import os
import sys

import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from corine_unet.data import load_data, clean_images_and_labels, normalize_data, split_data
from corine_unet.model import build_model
from corine_unet.train import build_callbacks, train_model, plot_training_metrics
from corine_unet.evaluate import calculate_test_metrics, plot_confusion_matrix


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config):
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    training = config["training"]
    lr = config["learning_rate"]

    # Load and clean the data
    images, labels2 = load_data(config["data"]["images_path"], config["data"]["labels_path"])
    images, labels2 = clean_images_and_labels(images, labels2)

    # Create and compile the model
    model = build_model(learning_rate=lr["optimizer"])

    # Normalize images and split
    normalized_images = normalize_data(images)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        normalized_images, labels2, seed=training["seed"],
        test_size=training["test_size"], val_size=training["val_size"]
    )

    callbacks = build_callbacks(
        os.path.join(output_dir, "best_model.keras"), lr, total_epochs=training["epochs"],
        early_stopping_patience=training["early_stopping_patience"]
    )
    history = train_model(
        model, X_train, y_train, X_val, y_val, callbacks,
        epochs=training["epochs"], batch_size=training["batch_size"]
    )

    # Plot training metrics
    fig = plot_training_metrics(history)
    plt.show()
    fig.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300, bbox_inches="tight")

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_test_metrics(y_test, y_pred_test)

    # Generate confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_test,
        save_path=os.path.join(output_dir, "confusion_matrix_percentage.png")
    )
    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the CORINE Level-2 U-Net model.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
