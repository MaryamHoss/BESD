# done, changed to correct

import h5py, os
import numpy as np
from numpy.random import seed

from tensorflow.python.keras.utils.data_utils import Sequence

'''if tf.__version__[:2] == '1.':
    from tensorflow import compat
    compat.v1.set_random_seed(14)
elif tf.__version__[:2] == '2.':
    tf.random.set_seed(14)'''


class Prediction_Generator(Sequence):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 sound_shape=(3, 1),
                 spike_shape=(3, 1),
                 batch_size=32,
                 data_type='',
                 downsample_sound_by=3):

        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             sound_shape=sound_shape,
                             spike_shape=spike_shape,
                             batch_size=batch_size,
                             data_type=data_type,
                             downsample_sound_by=downsample_sound_by
                             )

        self.input_len_1 = sound_shape[0]
        self.input_len_2 = spike_shape[0]
        self.output_len = spike_shape[0] if 'spike' in filepath_output else sound_shape[0]

        self.batch_size = batch_size
        self.batch_index = 0

        self.count_lines_in_file()
        self.on_epoch_end()
        self.ratio_sound_spike = int(sound_shape[0] / downsample_sound_by) / spike_shape[0]
        assert self.ratio_sound_spike.is_integer()
        self.select_subject(None)

    def __len__(self):
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input_first, 'r')
        for key in f.keys():
            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        return self.batch_generation()

    def select_subject(self, subject=2):
        self.samples_of_interest = range(self.input_file_first[self.input_1_key].shape[
                                             0])  # this takes the range of the number of test samples in the data
        if not subject == None:  # if we have a subject
            head, tail = os.path.split(self.filepath_output)  # separates the folder from the file name
            set = [s for s in ['train', 'val', 'test'] if s in tail][
                0]  # takes the name of the test data (clean_test.h5)
            subject_path = os.path.join(*[head, 'subjects_{}.h5'.format(
                set)])  # adds the subjects_test.h5 to the end of the path, so it gives us the subject list path
            subject_file = h5py.File(subject_path, 'r')
            subject_key = [key for key in subject_file][0]
            self.samples_of_interest = [i for i, s in enumerate(subject_file[subject_key][:]) if s == subject]
            # takes the list of subjects, finds all the indexes of the array items that correspons to the subject
        self.nb_lines = len(
            self.samples_of_interest)  # returns the number of the samples in the test data that correspond to the subject of interest
        # if the method select_subject with a subject of interest is not called, this will return all the samples of the data, no problem for train and validation

    def on_epoch_end(self):
        self.batch_index = 0
        self.input_file_first = h5py.File(self.filepath_input_first, 'r')
        self.input_file_second = h5py.File(self.filepath_input_second, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

        self.input_1_key = [key for key in self.input_file_first][0]
        self.input_2_key = [key for key in self.input_file_second][0]
        self.output_key = [key for key in self.output_file][0]

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size

        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        samples = self.samples_of_interest[
                  batch_start:batch_stop]  # if batch=1, takes each samples index belonging to the subject of interest
        input_batch_first = self.input_file_first[self.input_1_key][samples, :self.input_len_1]  # and load them
        input_batch_second = self.input_file_second[self.input_2_key][samples, :self.input_len_2]
        output_batch = self.output_file[self.output_key][samples, 1:self.output_len + 1]

        input_batch_first = input_batch_first[:, ::self.downsample_sound_by]
        input_batch_second = np.repeat(input_batch_second, self.ratio_sound_spike, 1)
        output_batch = output_batch[:, ::self.downsample_sound_by]

        # print(input_batch_first.shape, input_batch_second.shape, output_batch.shape)
        if 'noSpikes' in self.data_type:
            return input_batch_first, output_batch
        elif 'WithSpikes' in self.data_type:
            return [input_batch_first, input_batch_second], output_batch
        else:
            raise NotImplementedError


class Subject_Prediction_Generator(Prediction_Generator):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 sound_shape=(3, 1),
                 spike_shape=(3, 1),
                 batch_size=32,
                 data_type='',
                 downsample_sound_by=3):

        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             sound_shape=sound_shape,
                             spike_shape=spike_shape,
                             batch_size=batch_size,
                             data_type=data_type,
                             downsample_sound_by=downsample_sound_by
                             )

        self.input_len_1 = sound_shape[0]
        self.input_len_2 = spike_shape[0]
        self.output_len = spike_shape[0] if 'spike' in filepath_output else sound_shape[0]

        self.batch_size = batch_size
        self.batch_index = 0

        self.count_lines_in_file()
        self.on_epoch_end()

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.batch_index = 0
        self.input_file_first = h5py.File(self.filepath_input_first, 'r')
        self.input_file_second = h5py.File(self.filepath_input_second, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

        try:
            head, tail = os.path.split(self.filepath_output)
            set = [s for s in ['train', 'val', 'test'] if s in tail][0]
            self.subject_file = os.path.join(*[head, 'subjects_{}.h5'.format(set)])
        except:
            pass

        self.input_1_key = [key for key in self.input_file_first][0]
        self.input_2_key = [key for key in self.input_file_second][0]
        self.output_key = [key for key in self.output_file][0]

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size

        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        input_batch_first = self.input_file_first[self.input_1_key][batch_start:batch_stop, :self.input_len_1]
        input_batch_second = self.input_file_second[self.input_2_key][batch_start:batch_stop, :self.input_len_2]
        output_batch = self.output_file[self.output_key][batch_start:batch_stop, 1:self.output_len + 1]

        if 'noSpikes' in self.data_type:
            return input_batch_first[:, ::self.downsample_sound_by], \
                   output_batch[:, ::self.downsample_sound_by]
        elif 'WithSpikes' in self.data_type:
            return [input_batch_first[:, ::self.downsample_sound_by], input_batch_second], \
                   output_batch[:, ::self.downsample_sound_by]
        else:
            raise NotImplementedError


class Reconstruction_Generator(Prediction_Generator):
    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        input_batch_first = self.input_file_first[self.input_1_key][batch_start:batch_stop, :self.input_len_1]
        input_batch_second = self.input_file_second[self.input_2_key][batch_start:batch_stop, :self.input_len_2]
        output_batch = self.output_file[self.output_key][batch_start:batch_stop, 1:self.output_len + 1]

        return [input_batch_first[:, :, np.newaxis], input_batch_second[:, :, np.newaxis]], \
               output_batch[:, :, np.newaxis]


class Random_Generator(Sequence):
    def __init__(self,
                 sound_shape=None,
                 spike_shape=None,
                 batch_size=32,
                 data_type='noSpikes',
                 downsample_sound_by=3):

        self.__dict__.update(batch_size=batch_size, data_type=data_type, downsample_sound_by=downsample_sound_by)

        if 'WithSpikes' in data_type:
            self.input_shape = spike_shape
            self.output_shape = sound_shape
        elif 'noSpike' in data_type:
            self.input_shape = sound_shape
            self.output_shape = sound_shape
        else:
            raise NotImplementedError

    def __len__(self):
        return 2

    def __getitem__(self, index=0):
        return self.batch_generation()

    def batch_generation(self):
        input_batch = np.array(np.random.rand(self.batch_size, *self.input_shape), dtype='float32')
        output_batch = np.array(np.random.rand(self.batch_size, *self.output_shape), dtype='float32')

        if 'noSpike' in self.data_type:
            output = (input_batch[:, ::self.downsample_sound_by, :],
                      output_batch[:, ::self.downsample_sound_by, :])
        elif 'WithSpikes' in self.data_type:
            output = ([output_batch[:, ::self.downsample_sound_by, :], input_batch],
                      output_batch[:, ::self.downsample_sound_by, :])
        else:
            raise NotImplementedError

        return output


if __name__ == '__main__':
    gen = Subject_Prediction_Generator
