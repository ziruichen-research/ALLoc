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
        return csi, pos
    
    raw_dataset_paths = [f'./dichasus-ca01_part{k + 1}.tfrecords' for k in range(4)]
    offset_path = './reftx-offsets-dichasus-ca0x.json'
    with open(offset_path, 'r') as file:
        offsets = json.load(file)
    def apply_calibration(csi, pos):
        sto_offset = tf.tensordot(tf.constant(offsets['sto']), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets['cpo']), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
        return csi, pos
    
    raw_dataset = tf.data.TFRecordDataset(raw_dataset_paths).map(record_parse_function)
    raw_dataset = raw_dataset.map(apply_calibration)
    data_list = []
    loc_list = []
    downsampling_factor = 32
    ftype = 'fir'
    for csi, pos in raw_dataset:
        data = csi.numpy().astype(np.complex64) # [32, 1024]
        loc = pos.numpy().astype(np.float32) # [3]
        data_real = sp.signal.decimate(data.real, downsampling_factor, axis=1, zero_phase=True, ftype=ftype) # [32, 32]
        data_imag = sp.signal.decimate(data.imag, downsampling_factor, axis=1, zero_phase=True, ftype=ftype) # [32, 32]
        data = data_real + 1j * data_imag # [32, 32]
        assert tuple(data.shape) == (32, 32)
        assert tuple(loc.shape) == (3,)
        data_list.append(data)
        loc_list.append(loc)
    data = np.stack(data_list, axis=0) # [num_data, 32, 32]
    data_loc = np.stack(loc_list, axis=0) # [num_data, 3]
    print(data.shape, data_loc.shape) # (116701, 32, 32) (116701, 3)
    print(np.mean(np.abs(data) ** 2)) # 9.048489
    
    assignments = [
        [0,13,31,29,3,7,1,12],
        [30,26,21,25,24,8,22,15],
        [28,5,10,14,6,2,16,18],
        [19,4,23,17,20,11,9,27]
    ]
    assignments = np.ravel(assignments)
    
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    num_car = 32
    num_ant = 32
    num_train_data = 40000
    train_data_start = 0
    num_test_data = 20000
    test_data_start = 80000
    
    data_dir = f'../dataset_dichasus/ca01/'
    os.makedirs(data_dir, exist_ok=True)

    indices = np.arange(data_loc.shape[0])
    np.random.shuffle(indices)
    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(data_dir, f'{dataset_type}.h5')
        if os.path.exists(dataset_path):
            continue
        num_data = eval(f'num_{dataset_type}_data')
        data_start = eval(f'{dataset_type}_data_start')
        dataset = h5py.File(dataset_path, 'w')
        data_dataset = dataset.create_dataset('data',
                                              (num_data, num_ant, num_car),
                                              maxshape=(None, num_ant, num_car),
                                              chunks=(1, num_ant, num_car),
                                              dtype='complex64')
        loc_dataset = dataset.create_dataset('data_loc', (num_data, 3), dtype='float32')
        for i in range(num_data):
            j = indices[i + data_start]
            data_dataset[i] = data[j, assignments]
            loc_dataset[i] = data_loc[j]
        dataset.close()