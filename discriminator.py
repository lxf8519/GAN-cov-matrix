import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name, is_training, num_ant=32, layers=2):
    self._name = name
    self._is_training = is_training
    self._num_ant = num_ant
    self._layers = layers
    self._reuse = False

  def __call__(self, x, y1):
    """
    Args:
      x: The covariance matrix, batch_size x M x M x layers
      y1: Auxiliary info for conditional GAN which is the omni training signal in the paper,
          reshaped and duplicated to batch_size x 4 x 4 x 512 for concatenation
    Returns:
      output: Whether x is from G, batch_size x 1 x 1 x 1
    """
    with tf.variable_scope(self._name, reuse=self._reuse):
      # convolution layers
      conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
      lrelu1 = ops.lrelu(conv1, 0.2)

      # 2nd hidden layer
      conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=(2, 2), padding='same')
      lrelu2 = ops.lrelu(tf.layers.batch_normalization(conv2, training=self._is_training), 0.2)

      # 3rd hidden layer
      conv3 = tf.layers.conv2d(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
      lrelu3 = ops.lrelu(tf.layers.batch_normalization(conv3, training=self._is_training), 0.2)
      # lrelu3 is 4x4x1024 from input 32x32, y1 is 1x128, replicate y1 to 4x4x128, then concatenate lrelu3 and y1 to make a 4x4x1152
      # 4th hidden layer
      if self._num_ant == 64:
        conv4 = tf.layers.conv2d(lrelu3, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = ops.lrelu(tf.layers.batch_normalization(conv4, training=self._is_training), 0.2)
      else:
        lrelu4 =  lrelu3

      conditional_conv = tf.concat(axis=3, values=[lrelu4, y1])
      # Use a [1,1] conv layer+Relu before the output layer, strides 1,1
      conv4_1 = tf.layers.conv2d(conditional_conv, 512, [1, 1], strides=(1, 1), padding='same')
      relu4_1=tf.nn.relu(conv4_1)
      # output layer
      conv5 = tf.layers.conv2d(relu4_1, 1, [4, 4], strides=(1, 1), padding='valid')
      output = tf.nn.sigmoid(conv5)

    self._reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

    return output