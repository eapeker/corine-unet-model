import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam

from .losses import combined_loss, custom_MeanIoU


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


def build_model(learning_rate=0.001):
    """Create and compile the model"""
    model = create_advanced_unet_model()

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=['accuracy', custom_MeanIoU(num_classes=15)]
    )
    return model
