import math
import time
from pathlib import Path
from fractions import Fraction

# scientific
import numpy as np
import beatbrain
from beatbrain import utils
from beatbrain.model.layers import (
    ConvBlock2D,
    DownSample2D,
    UpSample2D,
    ReconstructionLoss,
    KLLoss,
)
from beatbrain.model import helpers

# visualization
from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf

from tensorflow.keras import Model, Sequential, Input, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

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
    Subtract,
    Multiply,
    BatchNormalization,
    ReLU,
    Activation,
)
from tensorflow.keras.callbacks import (
    Callback,
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN,
    CSVLogger,
    LambdaCallback,
)


def _build_encoder(input_shape, latent_dim, repeat, use_inception):
    def reparam(args):
        z_mean, z_log_var = args
        dim = tf.keras.backend.int_shape(z_mean)[1]
        eps = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        #         eps = tf.keras.backend.random_normal(shape=(batch_size, dim))
        return eps * tf.exp(z_log_var * 0.5) + z_mean

    encoder_input = Input(
        shape=input_shape,
        #         batch_size=batch_size,
        name="encoder_input",
    )
    e = Conv2D(32, 1)(encoder_input)
    e = ConvBlock2D(32, repeat, use_inception=use_inception, transpose=False)(e)
    e = DownSample2D(64, 4)(e)
    e = ConvBlock2D(64, repeat, use_inception=use_inception, transpose=False)(e)
    e = DownSample2D(128, 4)(e)
    e = ConvBlock2D(128, repeat, use_inception=use_inception, transpose=False)(e)
    e = DownSample2D(256, 2)(e)
    e = ConvBlock2D(256, repeat, use_inception=use_inception, transpose=False)(e)
    e = AveragePooling2D(8)(e)
    e = Flatten()(e)
    z_mean = Dense(latent_dim, name="z_mean")(e)
    z_log_var = Dense(latent_dim, name="z_log_var")(e)
    z = Lambda(reparam, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    encoder = Model(
        inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="encoder"
    )
    return encoder_input, encoder


def _build_decoder(latent_dim, output_shape, repeat, use_inception):
    decoder_input = Input(shape=latent_dim, name="decoder_input")
    # TODO: Use `fractions.Fraction` to calculate reduced fraction and upsample accordingly
    start_shape = [
        None,
        output_shape[0] // 32,
        output_shape[1] // 32,
        output_shape[2],
    ]
    d = Dense(start_shape[1] * start_shape[2] * start_shape[3], activation="relu",)(
        decoder_input
    )
    d = Reshape(target_shape=(start_shape[1], start_shape[2], start_shape[3],))(d)
    d = ConvBlock2D(256, repeat, use_inception, transpose=True)(d)
    d = UpSample2D(128, 2)(d)
    d = ConvBlock2D(128, repeat, use_inception, transpose=True)(d)
    d = UpSample2D(64, 4)(d)
    d = ConvBlock2D(64, repeat, use_inception, transpose=True)(d)
    d = UpSample2D(32, 4)(d)
    d = ConvBlock2D(32, repeat, use_inception, transpose=True)(d)
    d = Conv2DTranspose(1, 1)(d)
    d = Activation(tf.nn.sigmoid)(d)
    decoder = Model(inputs=decoder_input, outputs=d, name="decoder")
    return decoder_input, decoder


def build(
    latent_dim,
    input_shape,
    repeat=1,
    use_inception=True,
    batch_size=1,
    learning_rate=1e-4,
):
    encoder_input, encoder = _build_encoder(
        input_shape, latent_dim, repeat, use_inception
    )
    decoder_input, decoder = _build_decoder(
        latent_dim, input_shape, repeat, use_inception
    )
    z_mean, z_log_var, z = encoder(encoder_input)
    decoder_output = decoder(z)
    model = Model(encoder_input, decoder_output, name="vae")

    print(f"Encoder input: {encoder_input.shape}")
    print(f"Decoder output: {decoder_output.shape}")
    encoder_input.shape.assert_is_compatible_with(decoder_output.shape)
    #     assert encoder_input.shape.as_list() == decoder_output.shape.as_list()

    reconstruction_loss = ReconstructionLoss(mean=True)([encoder_input, decoder_output])
    #     reconstruction_loss = tf.losses.mse(encoder_input, decoder_output)
    #     reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[1, 2])
    kl_loss = KLLoss(mean=True)([z, z_mean, z_log_var])
    #     logpz = log_normal_pdf(z, 0.0, 0.0)
    #     logqz_x = log_normal_pdf(z, z_mean, z_log_var)
    #     kl_loss = logqz_x - logpz
    vae_loss = reconstruction_loss + kl_loss
    model.add_loss(vae_loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=lambda yt, yp: vae_loss)

    model.add_metric(
        reconstruction_loss, aggregation="mean", name="reconstruction_loss"
    )
    model.add_metric(kl_loss, aggregation="mean", name="kl_loss")
    return model, encoder, decoder
