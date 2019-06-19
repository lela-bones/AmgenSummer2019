from abc import ABCMeta, abstractmethod
import h5py
from numpy import array, tile, append
from scipy.io import loadmat


def load_from_mat(input_filename, name):
    if input_filename[-4:] != ".mat":
        input_filename += ".mat"
    var_mat = loadmat(input_filename, squeeze_me=True)
    var = var_mat[name]
    return var


class FeatReader(metaclass=ABCMeta):
    """
    Abstract class for reading features extracted from datasets.
    To add a new feature for a dataset, extract features for each frame and
    each video sequence, save them in feat_files whose names include the
    annotation.video_name and add a suitable FeatReader here.
    """
    def __init__(self):
        # TODO: add parameter for downsampling_factor
        pass

    @abstractmethod
    def read_feat(self, feat_file, downsampling_factor=1, segment=None):
        """
        Reads feature array from feat_file.
        :param feat_file:  full path to file with features for one sequence
        :param downsampling_factor: video duration downsampling factor
        :param segment: (start_fr, end_fr) tuple refering to temporal segment
                        AFTER any downsampling
        :return: feat: numpy array with shape (nb_timesteps, feat_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        """
        Reads labels from feat_file
        :param feat_file:  full path to file with features for one sequence
               nb_classes: number of classes
        :param nb_classes: number of action classes
        :param downsampling_factor: video duration downsampling factor
        :param segment: (start_fr, end_fr) tuple refering to temporal segment
                        AFTER any downsampling

        :return: labels: numpy array with shape (nb_timesteps) containing
                         contiguous action labels [0, C-1]
        """
        raise NotImplementedError

    @abstractmethod
    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):
        """
        Reads frame indices from feat_file
        :param feat_file:  full path to file with features for one sequence
        :param downsampling_factor: video duration downsampling factor
        :param segment: (start_fr, end_fr) tuple refering to temporal segment
                        AFTER any downsampling

        :return: frame_indices: numpy array with shape (nb_timesteps)
        """
        raise NotImplementedError


class LeaSpatialCNNFeatReader(FeatReader):
    def __init__(self):
        super().__init__()

    def read_feat(self, feat_file, downsampling_factor=1, segment=None):
        feat = load_from_mat(feat_file, 'A')
        feat = feat[::downsampling_factor, :]
        if segment is not None:
            feat = feat[segment[0]:segment[1]+1, :]
        return feat

    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        labels = load_from_mat(feat_file, 'Y')
        labels = labels[::downsampling_factor]
        if segment is not None:
            labels = labels[segment[0]:segment[1]+1]
        return labels

    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):
        frame_indices = None
        return frame_indices


class LeaSpatialCNNPredictionsReader(FeatReader):
    def __init__(self):
        super().__init__()

    def read_feat(self, feat_file, downsampling_factor=1, segment=None):
        feat = load_from_mat(feat_file, 'X')
        feat = feat[::downsampling_factor, :]
        if segment is not None:
            feat = feat[segment[0]:segment[1]+1, :]
        return feat

    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        labels = load_from_mat(feat_file, 'Y')
        labels = labels[::downsampling_factor]
        if segment is not None:
            labels = labels[segment[0]:segment[1]+1]
        return labels

    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):
        frame_indices = None
        return frame_indices


class KuehneIDTFVHDF5FeatReader(FeatReader):
    def __init__(self):
        super().__init__()

    def read_feat(self, feat_file, downsampling_factor=1,
                  segment=None):
        # frames: numpy array nb_frames x 1
        # feat: numpy array: nb_frames x feat_dim

        # Read hdf5 file
        # dt_l2pn_c64_pc64: dense trajectories, l2 power normalization
        # 426 -> 64 dim for feature descriptor
        # x -> 64 dim for fisher vector
        with h5py.File(feat_file, 'r') as hf:
            feat = array(hf['feat'])
        feat = feat[::downsampling_factor, :]
        if segment is not None:
            feat = feat[segment[0]:segment[1]+1, :]
        return feat

    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        # Files do not contain labels, get them from annotations
        return None

    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):
        with h5py.File(feat_file, 'r') as hf:
            frame_indices = array(hf['frame_ind'])
        frame_indices = frame_indices[::downsampling_factor]
        if segment is not None:
            frame_indices = frame_indices[segment[0]:segment[1]+1]
        return frame_indices


class SigTwoStreamHDF5FeatReader(FeatReader):
    def __init__(self, pad_feat_len=-1, pad_mode='repeat_last'):
        """
        :param pad_feat_len: pad original features, before any downsampling,
                so that they are of shape (nb_frames+pad_len, feat_dim)
        :param pad_mode: how to pad, zero_feat or repeat_last
        """
        super().__init__()
        self.pad_feat_len = pad_feat_len
        self.pad_mode = pad_mode

    def read_feat(self, feat_file, downsampling_factor=1,
                  segment=None):
        """

        :param feat_file: hdf5 feature file with fields feat, frame_ind
        :param downsampling_factor: downsampling factor, multiple of 4, because
                                    feat are extracted every 4 frames.
        :param segment:
        :return:
        """

        # frames: numpy array nb_frames x 1
        # feat: numpy array: nb_frames x feat_dim

        # downsampling is already one every 4 frames
        if downsampling_factor % 4 != 0:
            raise ValueError('Downsampling factor should be a multiple of 4')

        downsampling_factor = int(downsampling_factor // 4)

        # Read hdf5 file
        # feat_dim: 4096
        # frame_indices: starting from 0, feat available every 4 frames
        with h5py.File(feat_file, 'r') as hf:
            feat = array(hf['feat'])
        if self.pad_feat_len != -1:
            if self.pad_mode == 'repeat_last':
                feat = append(feat, tile(feat[-1, :], (self.pad_feat_len, 1)),
                              axis=0)
            else:
                raise ValueError('Invalid pad mode')

        feat = feat[::downsampling_factor, :]
        if segment is not None:
            feat = feat[segment[0]:segment[1]+1, :]
        return feat

    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        # Files do not contain labels, get them from annotations
        return None

    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):

        if downsampling_factor % 4 != 0:
            raise ValueError('Downsampling factor should be a multiple of 4')
        downsampling_factor = int(downsampling_factor // 4)

        with h5py.File(feat_file, 'r') as hf:
            frame_indices = array(hf['frame_ind'])

        frame_indices = frame_indices[::downsampling_factor]
        if segment is not None:
            frame_indices = frame_indices[segment[0]:segment[1]+1]
        return frame_indices


class PiergiajI3DHDF5FeatReader(FeatReader):
    def __init__(self, pad_feat_len=-1, pad_mode='repeat_last'):
        """
        :param pad_feat_len: pad original features, before any downsampling,
                so that they are of shape (nb_frames+pad_len, feat_dim)
        :param pad_mode: how to pad, zero_feat or repeat_last
        """
        super().__init__()
        self.pad_feat_len = pad_feat_len
        self.pad_mode = pad_mode

    def read_feat(self, feat_file, downsampling_factor=1,
                  segment=None):
        """

        :param feat_file: hdf5 feature file with fields feat, frame_ind
        :param downsampling_factor: downsampling factor, multiple of 8, because
                                    feat are extracted every 8 frames.
        :param segment:
        :return:
        """

        # frames: numpy array nb_frames x 1
        # feat: numpy array: nb_frames x 1024

        # downsampling is already one every 8 frames
        if downsampling_factor % 8 != 0:
            raise ValueError('Downsampling factor should be a multiple of 8')

        downsampling_factor = int(downsampling_factor // 8)

        # Read hdf5 file
        # feat_dim: 1024
        # frame_indices: starting from 0, feat available every 8 frames
        with h5py.File(feat_file, 'r') as hf:
            feat = array(hf['feat'])
        if self.pad_feat_len != -1:
            if self.pad_mode == 'repeat_last':
                feat = append(feat, tile(feat[-1, :], (self.pad_feat_len, 1)),
                              axis=0)
            else:
                raise ValueError('Invalid pad mode')

        feat = feat[::downsampling_factor, :]
        if segment is not None:
            feat = feat[segment[0]:segment[1]+1, :]
        return feat

    def read_labels(self, feat_file, nb_classes, downsampling_factor=1,
                    segment=None):
        # Files do not contain labels, get them from annotations
        return None

    def read_frame_indices(self, feat_file, downsampling_factor=1,
                           segment=None):

        if downsampling_factor % 8 != 0:
            raise ValueError('Downsampling factor should be a multiple of 8')
        downsampling_factor = int(downsampling_factor // 8)

        with h5py.File(feat_file, 'r') as hf:
            frame_indices = array(hf['frame_ind'])

        frame_indices = frame_indices[::downsampling_factor] - 1
        if segment is not None:
            frame_indices = frame_indices[segment[0]:segment[1]+1]
        return frame_indices
