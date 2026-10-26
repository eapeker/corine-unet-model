import os

import numpy as np

from main import load_config, run


def make_fake_data(directory, n_patches=20):
    """Random 6-band patches; each label patch is filled with one of two classes."""
    rng = np.random.default_rng(0)
    images = rng.uniform(0, 1000, size=(n_patches, 256, 256, 6)).astype(np.float32)
    labels = np.zeros((n_patches, 256, 256, 15), dtype=np.float32)
    labels[: n_patches // 2, :, :, 0] = 1.0
    labels[n_patches // 2 :, :, :, 1] = 1.0

    images_path = os.path.join(directory, "np_img.npy")
    labels_path = os.path.join(directory, "np_level_2.npy")
    np.save(images_path, images)
    np.save(labels_path, labels)
    return images_path, labels_path


def test_one_epoch_end_to_end(tmp_path):
    images_path, labels_path = make_fake_data(str(tmp_path))
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    config["data"] = {"images_path": images_path, "labels_path": labels_path}
    config["output_dir"] = str(tmp_path / "outputs")
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2  # small batch to keep CPU memory low

    history, test_metrics = run(config)

    assert len(history.history["loss"]) == 1
    # plot_training_metrics reads these metric names from the history
    assert "custom__mean_io_u" in history.history, sorted(history.history)
    assert "val_custom__mean_io_u" in history.history, sorted(history.history)
    assert len(test_metrics) == 15
    for name in ("best_model.keras", "training_metrics.png", "confusion_matrix_percentage.png"):
        assert os.path.exists(os.path.join(config["output_dir"], name))
