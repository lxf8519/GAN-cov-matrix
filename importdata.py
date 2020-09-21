import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    #labels_dense = labels_dense - 1
    #index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    for i in range(num_labels):
        labels_one_hot[i][int(labels_dense[i])] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self,
                 rg,
                 labels,
                 omni):
        """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
        self._num_examples = rg.shape[0]
        self._rg = rg
        self._labels = labels
        self._omni=omni
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def rg(self):
        return self._rg

    @property
    def labels(self):
        return self._labels

    @property
    def omni(self):
        return self._omni

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._rg = self.rg[perm0]
            self._labels = self.labels[perm0]
            self._omni = self._omni[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            rg_rest_part = self._rg[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            omni_rest_part = self._omni[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._rg = self.rg[perm]
                self._labels = self.labels[perm]
                self._omni = self.omni[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            rg_new_part = self._rg[start:end]
            labels_new_part = self._labels[start:end]
            omni_new_part = self._omni[start:end]
            return np.concatenate((rg_rest_part, rg_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0),np.concatenate(
                (omni_rest_part, omni_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._rg[start:end], self._labels[start:end], self._omni[start:end]


def read_data_sets(filename_db_tr = 'Datasets/ML_input_street_LOS_training.mat',
                   filename_db_test='Datasets/ML_input_street_LOS_test.mat',
                   db_size_percent = 1,
                   HDF5file=True):

    if not HDF5file:
        rtmp = scio.loadmat(filename_db_tr)
        train_rg = rtmp['R_input_training']
        train_omni = rtmp['omni_input_training']
        rtmp = scio.loadmat(filename_db_test)
        test_rg = rtmp['R_input_test']
        test_omni = rtmp['omni_input_test']
    else:
        file = h5py.File(filename_db_tr)
        train_rg = np.array(file['R_input_training']).transpose()
        train_omni = np.array(file['omni_input_training']).transpose()
        file = h5py.File(filename_db_test)
        test_rg = np.array(file['R_input_test']).transpose()
        test_omni = np.array(file['omni_input_test']).transpose()

    num_samples = int(db_size_percent * train_rg.shape[0])
    train_labels = np.arange(0, num_samples)
    if db_size_percent == 1:
        train_rg_subset = train_rg
        train_omni_subset = train_omni
    else:
        idx  = np.random.randint(0, train_rg.shape[0], size=(num_samples))
        train_rg_subset = train_rg[idx, :]
        train_omni_subset = train_omni[idx, :]
    del train_rg


    print('Dataset size is %d' % (num_samples))
    num_samples = test_omni.shape[0]
    test_labels = np.arange(0, num_samples)

    train = DataSet(train_rg_subset, train_labels, train_omni_subset)
    validation = DataSet(test_rg, test_labels,test_omni)

    return train, validation
