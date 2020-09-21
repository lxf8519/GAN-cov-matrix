from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import ConditionalGAN
from datetime import datetime
import os, time
import logging
import ops
from importdata import read_data_sets
import numpy as np
import scipy.io as scio

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 256, 'batch size, default: 256')
tf.flags.DEFINE_integer('num_ant', 64, 'number of antennas, default: 32')
tf.flags.DEFINE_integer('layers', 1, 'real and imaginary parts, default: 2')
tf.flags.DEFINE_integer('omni_size', 512, 'length of omni training sequence, default: 512')
tf.flags.DEFINE_integer('z_size', 100, 'length of random vector z')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_string('filename_db_tr', 'Datasets/ML_input_street_LOS_training_n21.mat',
                       'file for training, default: num_ant=32, 50k samples')
tf.flags.DEFINE_string('filename_db_test', 'Datasets/ML_input_street_LOS_test_n21.mat',
                       'file for testing, default: num_ant=32, 1k samples')
tf.flags.DEFINE_float('db_size_percent', 1, 'Percentage of dataset used in training, default: 1')
tf.flags.DEFINE_bool('HDF5file', False, 'Dataset file is HDF5 format, default: True')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    cgan = ConditionalGAN(
        batch_size=FLAGS.batch_size,
        num_ant=FLAGS.num_ant,
        layers=FLAGS.layers,
        learning_rate=FLAGS.learning_rate,
        omni_size=FLAGS.omni_size,
        z_size=FLAGS.z_size
    )
    G_loss, D_loss, fake_x, mse = cgan.model()
    optimizersG = cgan.G_optimize(G_loss)
    optimizersD = cgan.D_optimize(D_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()
  train, testset = read_data_sets(FLAGS.filename_db_tr, FLAGS.filename_db_test, FLAGS.db_size_percent, FLAGS.HDF5file)
  test_size = testset.omni.shape[0]
  test_nmse = 1

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

      while not coord.should_stop():
        # train
        rg, _, omni = train.next_batch(FLAGS.batch_size)
        if FLAGS.layers == 2:
          rg_r = np.reshape(rg, [FLAGS.batch_size, FLAGS.num_ant, FLAGS.num_ant, FLAGS.layers])
        else:
          rg_r = rg[:, ::2]
          rg_r = np.reshape(rg_r, [-1, FLAGS.num_ant, FLAGS.num_ant, 1])
        omni_real_reshaped = ops.omni_reshape(omni, FLAGS.batch_size, FLAGS.omni_size)
        _, _, fake_omni = train.next_batch(FLAGS.batch_size)
        omni_fake_reshaped = ops.omni_reshape(fake_omni, FLAGS.batch_size, FLAGS.omni_size)
        omni = np.reshape(omni, [-1, 1, 1, FLAGS.omni_size])
        z_value = np.random.uniform(-1., 1., size=(FLAGS.batch_size, FLAGS.z_size)).astype(np.float32)
        z_value = np.reshape(z_value, [-1, 1, 1, FLAGS.z_size])

        _, D_loss_val, summary = (
              sess.run(
                  [optimizersD, D_loss, summary_op],
                  feed_dict={cgan.z: z_value, cgan.x: rg_r, cgan.y: omni,
                             cgan.y1: omni_real_reshaped, cgan.y2: omni_fake_reshaped}
              )
        )
        rg, _, omni = train.next_batch(FLAGS.batch_size)
        if FLAGS.layers == 2:
          rg_r = np.reshape(rg, [FLAGS.batch_size, FLAGS.num_ant, FLAGS.num_ant, FLAGS.layers])
        else:
          rg_r = rg[:, ::2]
          rg_r = np.reshape(rg_r, [-1, FLAGS.num_ant, FLAGS.num_ant, 1])
        omni_real_reshaped = ops.omni_reshape(omni, FLAGS.batch_size, FLAGS.omni_size)
        omni = np.reshape(omni, [-1, 1, 1, FLAGS.omni_size])
        z_value = np.random.uniform(-1., 1., size=(FLAGS.batch_size, FLAGS.z_size)).astype(np.float32)
        z_value = np.reshape(z_value, [-1, 1, 1, FLAGS.z_size])
        _, G_loss_val = (
          sess.run(
            [optimizersG, G_loss],
            feed_dict={cgan.z: z_value, cgan.x: rg_r, cgan.y: omni,
                       cgan.y1: omni_real_reshaped}
          )
        )
        train_writer.add_summary(summary, step)
        train_writer.flush()
        if step % 5 == 0:
          train_mse = mse.eval(feed_dict={cgan.x: rg_r, cgan.z: z_value, cgan.y: omni})
          z_value = np.random.uniform(-1., 1., size=(test_size, FLAGS.z_size)).astype(np.float32)
          z_value = np.reshape(z_value, [-1, 1, 1, FLAGS.z_size])
          omni_test = np.reshape(testset.omni, [-1, 1, 1, FLAGS.omni_size])
          samples = sess.run(fake_x, feed_dict={cgan.z: z_value, cgan.y: omni_test, cgan.is_training: False})
          if FLAGS.layers == 2:
            rg_test = np.reshape(testset.rg, [test_size, FLAGS.num_ant, FLAGS.num_ant, 2])
          else:
            rg_test = testset.rg[:, ::2]
            rg_test = np.reshape(rg_test, [test_size, FLAGS.num_ant, FLAGS.num_ant, 1])
          nmse = np.zeros([test_size, 1], dtype=np.float64)
          for i2 in range(test_size):
            nmse[i2] = np.sum(np.square(np.abs(rg_test[i2, :, :, :] - samples[i2, :, :, :]))) / np.sum(
              np.square(np.abs(rg_test[i2, :, :, :])))
          test_nmse = np.sum(nmse) / test_size
          if test_nmse < 0.0008:
            filename_db_save = 'Estimated_results/Estimated_cov_mat_%d_mse_%g.mat' % (step, test_nmse)
            scio.savemat(filename_db_save,
                         {'R_estimated': samples, 'saved_omni_test': testset.omni, 'test_nmse': test_nmse,
                          'D_loss': G_loss_val,
                          'G_loss': G_loss_val})
          print("step %d, training_MSE %g, test_NMSE %g, D_loss %g, G_loss %g" % (
          step, train_mse, test_nmse, D_loss_val, G_loss_val))

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_loss   : {}'.format(D_loss_val))

        if step % 10000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()