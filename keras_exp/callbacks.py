"""
"""
import os

from keras import backend as K
from keras.layers import Embedding

from ._mixin_common import mixedomatic

if K.backend() == 'tensorflow':
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector
    from keras.callbacks import TensorBoard


__all__ = ('TensorBoardEmbedding', 'find_embedding_layers', )


def find_embedding_layers(layers):
    '''Recursively find embedding layers.

    :param layers: The keras model layers. Typically obtained via model.layers
    :type layers: list
    '''
    elayers = []
    for layer in layers:
        if isinstance(layer, Embedding):
            elayers.append(layer)

        slayers = getattr(layer, 'layers', [])
        elayers += find_embedding_layers(slayers)

    return elayers


class TensorBoardEmbeddingMixin(object):
    """Tensorboard mixin for Embeddings.

    This has to be mixed in with TensorBoard class or a derived TensoBoard cls.
    Must specify arguments as keywords.

    # Mixin Arguments
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """
    def __init__(self,
                 embeddings_freq=1,
                 embeddings_layer_names=None,
                 embeddings_metadata={}):
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata

    def set_model(self, model):
        if self.embeddings_freq:
            self.saver = tf.train.Saver()

            embeddings_layer_names = self.embeddings_layer_names

            elayers = find_embedding_layers(model.layers)
            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in elayers]

            embeddings = {layer.name: layer.weights[0] for layer in elayers
                          if layer.name in embeddings_layer_names}

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_logs = []

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                self.embeddings_logs.append(os.path.join(self.log_dir,
                                                         layer_name + '.ckpt'))

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        if self.embeddings_freq and self.embeddings_logs:
            if epoch % self.embeddings_freq == 0:
                for log in self.embeddings_logs:
                    self.saver.save(self.sess, log, epoch)


@mixedomatic()
class TensorBoardEmbedding(TensorBoardEmbeddingMixin, TensorBoard):
    """Tensorboard for Embeddings.

    Refer to classes TensorBoardEmbedding and TensorBoardEmbeddingMixin for
    arguments. Must specify arguments as keywords i.e. kwargs to __init__.
    """

    def set_model(self, model):
        TensorBoard.set_model(self, model)
        TensorBoardEmbeddingMixin.set_model(self, model)

    def on_epoch_end(self, epoch, logs=None):
        TensorBoardEmbeddingMixin.on_epoch_end(self, epoch)
        TensorBoard.on_epoch_end(self, epoch, logs)

