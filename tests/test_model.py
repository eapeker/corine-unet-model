import numpy as np

from corine_unet.model import create_advanced_unet_model


def test_output_shape():
    model = create_advanced_unet_model()
    assert model.output_shape == (None, 256, 256, 15)

    prediction = model.predict(np.zeros((1, 256, 256, 6), dtype=np.float32), verbose=0)
    assert prediction.shape == (1, 256, 256, 15)
    # Softmax output sums to 1 for every pixel
    assert np.allclose(prediction.sum(axis=-1), 1.0, atol=1e-4)
