from numpy import ones, asarray, max, zeros, insert, append, array
from abc import ABCMeta, abstractmethod
import pdb


def flatten_list(nested_list):
    """
    Flatten list of lists.
    :param nested_list:
    :return:
    """
    return [item for sublist in nested_list for item in sublist]


class DataPreprocessor(metaclass=ABCMeta):
    """
    Abstract base class used to build new data preprocessors
    """
    def __init__(self, params):
        """
        :param params: a dictionary of parameters
        """
        self.params = params

    @abstractmethod
    def preprocess(self, features_lst, labels_lst):
        """
        Args:
        features_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps, feat_dim) numpy array
        labels_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps,) numpy array or list

        :return: feat, labels, sample_weight
        """
        raise NotImplementedError

    def set_max_nb_frames(self, max_nb_frames):
        self.params['max_nb_frames'] = max_nb_frames


class Seq2SeqDataPreprocessor(metaclass=ABCMeta):
    """
    Abstract base class used to build new data preprocessors for frame and
    segment level labels.
    """
    def __init__(self, params):
        """
        :param params: a dictionary of parameters
        """
        self.params = params

    def set_max_nb_frames(self, max_nb_frames):
        self.params['max_nb_frames'] = max_nb_frames

    def set_max_nb_segs(self, max_nb_segs):
        self.params['max_nb_segs'] = max_nb_segs
        # Sequence start/sequence end will be appended to each sequence of
        # segment labels
        self.params['max_nb_segs'] = self.params['max_nb_segs'] + 2

    @abstractmethod
    def preprocess(self, features_lst, frame_labels_lst,
                   seg_labels_lst, segs_lst):
        """
        Args:
        features_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps, feat_dim) numpy array
        frame_labels_lst: a list with nb_samples elements. Each element is a
             (nb_timesteps,) numpy array or list
        seg_labels_lst: a list with nb_samples elements. Each element is a
             (nb_segs,) numpy array or list
        segs_lst: a list with nb_samples elements. Each element is a
             (nb_segs,) list: [(,), (,), (,), (,)]

        :return: feat, frame_labels, frame_sample_weight,
                 frame_sequence_lengths, seg_labels, seg_sample_weight,
                 seg_sequence_lengths, segs
        """
        raise NotImplementedError


def to_categorical_for_hinge(y, nb_classes=None):
    """
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix for use with hinge loss
    :param y: list of labels (integers from 0 to nb_classes-1)
    :param nb_classes: number of  classes
    :return: binary class matrix
    """
    y = asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = max(y)+1

    Y = -ones((len(y), nb_classes))

    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def pad_sequences(sequences, max_len=None, dtype='float32', value=0.):
    """

    :param sequences: list of nb_samples elements. Each element is an array
                      of shape (timesteps, feat_dim)
    :param max_len: if not None, pad all sequences to length max_len
    :param dtype: resulting numpy array dtype
    :param value: padding value
    :return:
            padded_sequences:
                numpy array with shape (nb_samples, max_len, feat_dim)
            padding_mask:
                numpy binary array with shape (nb_samples, max_len, 1),
                having zeroes at padded indices and ones elsewhere
    """

    nb_samples = len(sequences)
    # Get feature dimension from the first sequence
    feat_dim = sequences[0].shape[1]
    timesteps_per_sample = [s.shape[0] for s in sequences]
    max_timesteps = max(timesteps_per_sample)

    if max_len is None:
        max_len = max_timesteps
    else:
        if max_len < max_timesteps:
            raise ValueError("Max_len: {} less than max_timesteps: {}".format(
                max_len, max_timesteps))

    res = (ones((nb_samples, max_len, feat_dim)) * value).astype(dtype)
    padding_mask = zeros((nb_samples, max_len)).astype(dtype)
    for idx, s in enumerate(sequences):
        res[idx, :timesteps_per_sample[idx], :] = s
        padding_mask[idx, :timesteps_per_sample[idx]] = 1

    padded_sequences = res

    return padded_sequences, padding_mask


def pad_sequences_batch(sequences_batch, max_len, dtype='float32', value=0.):
    """

    :param sequences_batch: list of nb_batches elements.
        Each element is an array of shape (batch_size, timesteps, feat_dim)
    :param max_len: if not None, pad all sequences to length max_len
    :param dtype: resulting numpy array dtype
    :param value: padding value
    :return:
            padded_sequences:
                numpy array with shape (nb_samples, max_len, feat_dim)
    """

    nb_batches = len(sequences_batch)
    # Get info from the first sequence batch
    tensor_ndims = sequences_batch[0].ndim
    nb_samples = sum([sequences_batch[i].shape[0] for i in range(nb_batches)])
    timesteps_per_batch = [s.shape[1] for s in sequences_batch]
    if tensor_ndims == 2:
        res = (ones((nb_samples, max_len)) * value).astype(dtype)
    elif tensor_ndims == 3:
        feat_dim = sequences_batch[0].shape[-1]
        res = (ones((nb_samples, max_len, feat_dim)) * value).astype(dtype)
    else:
        raise ValueError('Not supported tensor shape')

    cnt = 0
    for (batch_ind, seq_batch) in enumerate(sequences_batch):
        batch_size = sequences_batch[batch_ind].shape[0]
        for seq_ind in range(batch_size):
            s = sequences_batch[batch_ind][seq_ind]
            res[cnt, :timesteps_per_batch[batch_ind]] = s
            cnt += 1

    padded_sequences = res

    return padded_sequences


def add_special_tokens(seg_labels_lst, segs_lst, nb_classes,
                       remove_bg_from_segs=False):
    """

    Prepends sequence start symbol (0),
    appends sequence end symbol (nb_classes+1),
    Increments label symbols by 1.
    :param seg_labels_lst: list of nb_samples elements. Each element is an array
                      of shape (nb_segs,)
            segs_lst: list of nb_samples elements. Each element is a
                    (nb_segs,) list: [(,), (,), (,), (,)]
    :param nb_classes: number of action classes (before adding the
                                                 special symbols)
    :param remove_bg_from_segs: remove segs corresponding to background class
                                (assumes background class==0)
    :return:
           res_seg_labels_lst: list of nb_samples elements. Each element is
                               an array of shape (nb_segs + 2,)
           res_segs_lst: list of nb_samples elements. Each element is an array
                         of shape (nb_segs + 2, 2)
    """

    res_seg_labels_lst = []

    for seg_labels in seg_labels_lst:

        seg_labels = asarray(seg_labels, dtype='int32')

        if remove_bg_from_segs:
            # Assumes background class id == 0
            seg_labels = seg_labels[seg_labels > 0]

        res_seg_labels = seg_labels + 1
        # Prepend sequence start symbol: 0
        res_seg_labels = insert(res_seg_labels, 0, [0])
        # Append sequence end symbol: nb_classes + 1
        res_seg_labels = append(res_seg_labels, [nb_classes + 1])
        res_seg_labels_lst.append(res_seg_labels)

    res_segs_lst = []
    # TODO: fix segs[:,0,:], segs[:,-1,:]
    # (segs corresponding to sequence start, sequence end)
    for segs in segs_lst:
        res_segs = [[-1, -1]] + segs + [[-1, -1]]
        res_segs_lst.append(array(res_segs))

    return res_seg_labels_lst, res_segs_lst


def to_categorical_multilabel(labels_per_frame, nb_classes=None):
    """Converts a list of labels per frame to a binary class matrix.
    E.g. for use with sigmoid_cross_entropy_with_logits.
    # Arguments
        labels_per_frame: list of lists with
                          labels per frame to be converted into a matrix
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """

    nb_frames = len(labels_per_frame)
    if not nb_classes:
        nb_classes = max(flatten_list(labels_per_frame)) + 1

    categorical = zeros((nb_frames, nb_classes))
    for i in range(nb_frames):
        # Handle frames without labels
        if len(labels_per_frame[i]) > 0:
            categorical[i, labels_per_frame[i]] = 1

    return categorical
