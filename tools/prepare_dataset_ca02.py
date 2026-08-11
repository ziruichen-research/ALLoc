import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel('ERROR')

import json
import random
import numpy as np
import scipy as sp
import h5py

if __name__ == '__main__':
    feature_description = {
        'csi': tf.io.FixedLenFeature([], tf.string, default_value = ''),
        'gt-interp-age-tachy': tf.io.FixedLenFeature([], tf.float32, default_value = 0),
        'pos-tachy': tf.io.FixedLenFeature([], tf.string, default_value = ''),
        'snr': tf.io.FixedLenFeature([], tf.string, default_value = ''),
        'time': tf.io.FixedLenFeature([], tf.float32, default_value = 0),
    }
    def record_parse_function(proto):
        record = tf.io.parse_single_example(proto, feature_description)
        csi = tf.ensure_shape(tf.io.parse_tensor(record['csi'], out_type = tf.float32), (32, 1024, 2))
        csi = tf.signal.fftshift(csi, axes=1)
        csi = tf.complex(csi[:,:,0], csi[:,:,1])
        pos = tf.ensure_shape(tf.io.parse_tensor(record['pos-tachy'], out_type = tf.float64), (3))
        pos = tf.cast(pos, tf.float32)
        time = tf.ensure_shape(record['time'], ())
        time = tf.cast(time, tf.float32)
        return csi, pos, time
    
    raw_dataset_paths = ['./dichasus-ca02.tfrecords']
    offset_path = './reftx-offsets-dichasus-ca0x.json'
    with open(offset_path, 'r') as file:
        offsets = json.load(file)
    def apply_calibration(csi, pos, time):
        sto_offset = tf.tensordot(tf.constant(offsets['sto']), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets['cpo']), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
        return csi, pos, time
    
    raw_dataset = tf.data.TFRecordDataset(raw_dataset_paths).map(record_parse_function)
    raw_dataset = raw_dataset.map(apply_calibration)
    data_list = []
    loc_list = []
    time_list = []
    downsampling_factor = 32
    ftype = 'fir'
    for csi, pos, time in raw_dataset:
        data = csi.numpy().astype(np.complex64) # [32, 1024]
        loc = pos.numpy().astype(np.float32) # [3]
        time = time.numpy().astype(np.float32) # []
        data_real = sp.signal.decimate(data.real, downsampling_factor, axis=1, zero_phase=True, ftype=ftype) # [32, 32]
        data_imag = sp.signal.decimate(data.imag, downsampling_factor, axis=1, zero_phase=True, ftype=ftype) # [32, 32]
        data = data_real + 1j * data_imag # [32, 32]
        assert tuple(data.shape) == (32, 32)
        assert tuple(loc.shape) == (3,)
        assert len(tuple(time.shape)) == 0
        data_list.append(data)
        loc_list.append(loc)
        time_list.append(time[()])
    data = np.stack(data_list, axis=0) # [32234, 32, 32]
    data_loc = np.stack(loc_list, axis=0) # [32234, 3]
    time = np.asarray(time_list) # [32234]
    print(data.shape, data_loc.shape, time.shape)
    print(np.mean(np.abs(data) ** 2)) # 9.303631
    
    assignments = [
        [0,13,31,29,3,7,1,12],
        [30,26,21,25,24,8,22,15],
        [28,5,10,14,6,2,16,18],
        [19,4,23,17,20,11,9,27]
    ]
    assignments = np.ravel(assignments)

    num_car = 32
    num_ant = 32
    num_test_data = 10000
    step_size = data_loc.shape[0] // num_test_data
    indices = np.argsort(time)[::step_size]
    indices = indices[:num_test_data]
    data = data[indices]
    data_loc = data_loc[indices]
    time = time[indices]
    
    data_dir = f'../dataset_dichasus/ca02/'
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, 'test.h5')
    if not os.path.exists(dataset_path):
        dataset = h5py.File(dataset_path, 'w')
        dataset.create_dataset('data', data=data[:, assignments])
        dataset.create_dataset('data_loc', data=data_loc)
        dataset.close()
        npy_path = os.path.join(data_dir, 'time.npy')
        np.save(npy_path, time)