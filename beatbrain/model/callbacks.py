from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from beatbrain import display
from beatbrain import utils


class VisualizeCallback(Callback):
    def __init__(
        self,
        log_dir,
        latent_dim,
        validation_data,
        n_examples=4,
        random_vectors=None,
        heatmap=True,
        distribution=False,
        reconstruction=True,
        generation=True,
        frequency="epoch",
        verbose=False,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.latent_dim = latent_dim
        self.n_examples = n_examples
        self.cmap = "magma" if heatmap else "Greys"
        self.frequency = frequency
        self.verbose = verbose
        self.total_batch = 0
        self.distribution = distribution
        self.reconstruction = reconstruction
        self.generation = generation
        self.random_vectors = random_vectors or tf.random.normal(
            shape=[n_examples, latent_dim]
        )
        self.fig = plt.figure()
        self.samples = list(validation_data.unbatch().take(self.n_examples))

        self.recon_raw = self.log_dir / "raw" / "reconstructed"
        self.recon_png = self.log_dir / "png" / "reconstructed"
        self.gen_raw = self.log_dir / "raw" / "generated"
        self.gen_png = self.log_dir / "png" / "generated"

    def on_train_begin(self, logs=None):
        self.recon_raw.mkdir(exist_ok=True, parents=True)
        self.recon_png.mkdir(exist_ok=True, parents=True)
        self.gen_raw.mkdir(exist_ok=True, parents=True)
        self.gen_png.mkdir(exist_ok=True, parents=True)

    def _visualize_reconstruction(self, batch=None, epoch=None):
        if not self.reconstruction:
            return
        assert (batch is not None) or (epoch is not None)
        fig = plt.figure(self.fig.number)
        fig.set_size_inches(10, 4)
        for i, sample in enumerate(self.samples):
            fig.add_subplot(121)
            sample = sample[None, :]
            display.show_spec(
                utils.denormalize_spectrogram(sample[0, ..., 0].numpy()),
                title="Original",
                cmap=self.cmap,
            )
            fig.add_subplot(122)
            reconstructed = self.model(sample)
            display.show_spec(
                utils.denormalize_spectrogram(reconstructed[0, ..., 0].numpy()),
                title="Reconstructed",
                cmap=self.cmap,
            )
            fig.tight_layout()
            title = f"recon_{i + 1}@{'epoch' if epoch else 'batch'}_{epoch or batch}"
            fig.suptitle(title)
            fig.savefig(self.recon_png / f"{title}.jpg")
            utils.save_image(
                reconstructed[0, ..., 0].numpy(), self.recon_raw / f"{title}.exr",
            )
            fig.clear()

        # TODO: Move distribution plotting to its own function
        if self.distribution:
            fig = plt.figure(self.fig.number)
            fig.set_size_inches(5, 4)
            sns.distplot(
                reconstructed[0, ..., 0].numpy().flatten(), ax=fig.add_subplot(111)
            )
            plt.show()
            fig.clear()

    def _visualize_generation(self, batch=None, epoch=None):
        if not self.generation:
            return
        assert (batch is not None) or (epoch is not None)
        decoder = self.model.get_layer("decoder")
        generated = decoder(self.random_vectors)
        fig = plt.figure(self.fig.number)
        fig.set_size_inches(5, 4)
        for i, gen in enumerate(generated):
            gen = gen[None, :]
            title = f"gen_{i + 1}@{'epoch' if epoch else 'batch'}_{epoch or batch}"
            display.show_spec(
                utils.denormalize_spectrogram(gen[0, ..., 0].numpy()),
                title=title,
                cmap=self.cmap,
            )
            fig.tight_layout()
            fig.savefig(self.gen_png / f"{title}.jpg")
            utils.save_image(gen[0, ..., 0], self.gen_raw / f"{title}.exr")
            fig.clear()

    def on_epoch_begin(self, epoch, logs=None):
        if self.frequency == "epoch":
            self._visualize_reconstruction(epoch=epoch)
            self._visualize_generation(epoch=epoch)

    def on_train_batch_begin(self, batch, logs=None):
        if isinstance(self.frequency, int) and (self.total_batch % self.frequency == 0):
            self._visualize_reconstruction(batch=self.total_batch)
            self._visualize_generation(batch=self.total_batch)

    def on_train_batch_end(self, batch, logs=None):
        self.total_batch += 1
