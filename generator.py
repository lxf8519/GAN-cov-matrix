import tensorflow as tf
import ops

class Generator:
  def __init__(self, name, is_training, num_ant=32, layers=2):
    self._name = name
    self._reuse = False
    self._is_training = is_training
    self._num_ant = num_ant # Number of antennas at BS which is M in the paper
    self._layers = layers # Real and imaginary parts of the covariance matrix

  def __call__(self, z, y):
    """
    Args:
      z: random vector for GAN, batch_size x 1 x 1 x 100
      y: Auxiliary info for conditional GAN which is the omni-received signal in the paper, batch_size x 1 x 1 x 512
    Returns:
      output: the covariance matrix, M x M x layers
    """
    with tf.variable_scope(self._name, reuse=self._reuse):
      # conv layers
      inputs = tf.concat(axis=3, values=[z, y])

      # 1st hidden layer
      conv1 = tf.layers.conv2d_transpose(inputs, 512, [4, 4], strides=(1, 1), padding='valid')
      lrelu1 = ops.lrelu(tf.layers.batch_normalization(conv1, training=self._is_training), 0.2)  # (?, 4, 4, 512)

      # 2nd hidden layer
      conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
      lrelu2 = ops.lrelu(tf.layers.batch_normalization(conv2, training=self._is_training), 0.2)  # (?, 8, 8, 256)

      # 3rd hidden layer
      conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2), padding='same')
      lrelu3 = ops.lrelu(tf.layers.batch_normalization(conv3, training=self._is_training), 0.2)  # (?, 16, 16, 128)

      if self._num_ant == 64:
        conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = ops.lrelu(tf.layers.batch_normalization(conv4, training=self._is_training), 0.2)
      else:
        lrelu4 = lrelu3


      # output layer
      conv5 = tf.layers.conv2d_transpose(lrelu4, self._layers, [4, 4], strides=(2, 2), padding='same')
      # (?, 32, 32, layers)

      output = tf.nn.tanh(conv5)

    self._reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

    return output
