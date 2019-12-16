"""
Custom layers
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPool2D,
    AveragePooling2D,
    Dense,
    Lambda,
    Reshape,
    Flatten,
    Layer,
    concatenate,
    Add,
    Multiply,
    BatchNormalization,
    ReLU,
    Activation,
)


class NormConv2D(Layer):
    """
    Batch-normalized, ReLU-activated convolution or transpose convolution
    """

    def __init__(self, filters, kernel_size, transpose=False):
        self._kernel_size = kernel_size
        self._filters = filters
        self._transpose = transpose
        super().__init__()

    def build(self, input_shape):
        conv_layer = Conv2DTranspose if self._transpose else Conv2D
        # TODO: Try with VALID padding
        self._model = Sequential(
            [
                conv_layer(self._filters, self._kernel_size, padding="SAME",),
                BatchNormalization(),
                ReLU(),
            ]
        )
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters": self._filters,
                "kernel_size": self._kernel_size,
                "transpose": self._transpose,
            }
        )
        return config


class Inception2D(Layer):
    def __init__(self, filters, transpose=False):
        self._filters = filters
        self._transpose = transpose
        super().__init__()

    def build(self, input_shape):
        filters = self._filters
        inputs = Input(shape=input_shape[1:])
        bottleneck = NormConv2D(filters, 1, transpose=self._transpose)(inputs)

        conv1 = NormConv2D(filters, 1, transpose=self._transpose)(bottleneck)
        conv3 = NormConv2D(filters, 3, transpose=self._transpose)(bottleneck)
        conv5 = NormConv2D(filters, 5, transpose=self._transpose)(bottleneck)
        conv7 = NormConv2D(filters, 7, transpose=self._transpose)(bottleneck)
        pool3 = MaxPool2D(pool_size=3, strides=1, padding="SAME")(inputs)
        pool5 = MaxPool2D(pool_size=5, strides=1, padding="SAME")(inputs)
        merged = Add()([conv1, conv3, conv5, conv7, pool3, pool5])
        self._model = Model(inputs=inputs, outputs=merged)
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"filters": self._filters, "transpose": self._transpose,}
        )
        return config


class DownSample2D(Layer):
    def __init__(self, filters, pool_kernel_size):
        self._filters = filters
        self._pool_kernel_size = pool_kernel_size
        super().__init__()

    def build(self, input_shape):
        self._model = Sequential(
            [
                AveragePooling2D(self._pool_kernel_size),
                Conv2D(self._filters, 1),
                BatchNormalization(),
                ReLU(),
            ],
        )
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"filters": self._filters, "pool_kernel_size": self._pool_kernel_size,}
        )
        return config


class UpSample2D(Layer):
    def __init__(self, filters, pool_kernel_size):
        self._filters = filters
        self._pool_kernel_size = pool_kernel_size
        super().__init__()

    def build(self, input_shape):
        self._model = Sequential(
            [
                Conv2DTranspose(self._filters, 1, strides=self._pool_kernel_size),
                BatchNormalization(),
                ReLU(),
            ],
        )
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"filters": self._filters, "pool_kernel_size": self._pool_kernel_size,}
        )
        return config


class ConvBlock2D(Layer):
    def __init__(self, filters, repeat, use_inception=True, transpose=False):
        self._filters = filters
        self._repeat = repeat
        self._use_inception = use_inception
        self._transpose = transpose
        super().__init__()

    def build(self, input_shape):
        if self._use_inception:
            layers = [
                Inception2D(self._filters, transpose=self._transpose)
                for i in range(self._repeat)
            ]
        else:
            layers = [
                NormConv2D(self._filters, 3, transpose=self._transpose)
                for i in range(self._repeat)
            ]
        self._model = Sequential(layers)
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters": self._filters,
                "repeat": self._repeat,
                "use_inception": self._use_inception,
                "transpose": self._transpose,
            }
        )
        return config


class ReconstructionLoss(Layer):
    # TODO: subclass `Loss` instead
    def __init__(self, mean=True):
        self._mean = mean
        super().__init__()

    def _mse(self, x):
        return tf.reduce_sum(tf.losses.mse(x[0], x[1]), axis=[1, 2])

    def build(self, input_shape):
        self._model = Sequential([Lambda(self._mse)])
        if self._mean:
            self._model.add(Lambda(tf.reduce_mean))
        super().build(input_shape)

    def call(self, x, **kwargs):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"mean": self._mean,}
        )
        return config


class KLLoss(Layer):
    # TODO: subclass `Loss` instead
    def __init__(self, mean=True):
        self._mean = mean
        super().__init__()

    def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def _kld(self, x):
        return self._log_normal_pdf(x[0], x[1], x[2]) - self._log_normal_pdf(
            x[0], 0.0, 0.0
        )

    def build(self, input_shape):
        self._model = Sequential([Lambda(self._kld)])
        if self._mean:
            self._model.add(Lambda(tf.reduce_mean))
        super().build(input_shape)

    def call(self, x, **kwargs):
        z, z_mean, z_log_var = x
        return self._model([z, z_mean, z_log_var])

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"mean": self._mean,}
        )
        return config


class Reparameterize(Layer):
    def build(self, input_shape):
        print("Reparam Layer input shape:", input_shape)
        inputs = Input(shape=[x[1:] for x in input_shape])
        print("Reparam input shape:", inputs.shape)
        # TODO: Get rid of lambda expressions inside Lambda layers
        epsilon = Lambda(
            lambda x: tf.keras.backend.random_normal(shape=tf.shape(x[0]))
        )(inputs)
        print("Epsilon shape:", epsilon.shape)
        mean = Lambda(lambda x: x[0])(inputs)
        print("Mean shape:", mean.shape)
        var = Lambda(lambda x: tf.exp(x[1] * 0.5))(inputs)
        print("Var shape:", var.shape)
        reparam = Multiply()([epsilon, var])
        print("Mul shape:", reparam.shape)
        reparam = Add()([reparam, mean])
        print("Add shape:", reparam.shape)
        self._model = Model(inputs=inputs, outputs=reparam)
        print("Reparam output shape:", self._model.output_shape)
        super().build(input_shape)

    def call(self, x):
        z_mean, z_log_var = x
        output = self._model([z_mean, z_log_var])
        print("Call time shape:", output.shape)
        return output

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)
