import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lfilter
from copy import deepcopy
from entropy import *
import entropy
from load_lyh_data import  *
from scipy.signal import welch
from scipy.io import loadmat, savemat

# config = {
#     'fs_raw' : 250,  # 原始数据采样频率
#     'fs_down' : 125,  # 下采样频率 
#     't_of_trial' : 4, # 一个trial的时间, 单位：s
#     'start': 2, # trial有效点起始index
#     'end': 2+1000, # trial有效点结尾index
#     'channel_index_list' : np.arange(2, 2+14),
#     'file_label_dict' : {
#         "nothing1_orginal.npy": 3, "nothing2_orginal.npy": 3, "nothing3_orginal.npy": 3,
#         "left1_orginal.npy": 0, "left2_orginal.npy": 0, "left3_orginal.npy": 0,
#         "right1_orginal.npy": 1, "right2_orginal.npy": 1, "right3_orginal.npy": 1,
#         "leg1_orginal.npy": 2, "leg2_orginal.npy": 2, "leg3_orginal.npy": 2
#         }, # 字典类型, 为data_dict的key到label的映射
#     'n_Classes' : 4, #类别数目
#     'pre_preparation_save_path': '/mnt/workspace/Baseline/lyh_data/session1/Standard_input', # 预处理结果保存路径
#     'Bandpass_filter_params' : { 'lowcut' : 0.05, 'highcut' : 40, 'fs' : 125, 'order' : 2 }, # 带通滤波器参数
#     'Preprocessed_save_path' : "/mnt/workspace/Baseline/lyh_data/session1/Preprocessed_data",
#     'Feature_extracted_save_path' : '/mnt/workspace/Baseline/lyh_data/session1/Feature_extracted'
# }

config = {
    'fs_raw' : 250,  # 原始数据采样频率
    'fs_down' : 125,  # 下采样频率 
    't_of_trial' : 4, # 一个trial的时间, 单位：s
    'start': 0, # trial有效点起始index
    'end': 1000, # trial有效点结尾index
    'channel_index_list' : np.arange(22),
    'file_label_dict' : {
        "nothing1_orginal.npy": 3, "nothing2_orginal.npy": 3, "nothing3_orginal.npy": 3,
        "left1_orginal.npy": 0, "left2_orginal.npy": 0, "left3_orginal.npy": 0,
        "right1_orginal.npy": 1, "right2_orginal.npy": 1, "right3_orginal.npy": 1,
        "leg1_orginal.npy": 2, "leg2_orginal.npy": 2, "leg3_orginal.npy": 2
        }, # 字典类型, 为data_dict的key到label的映射
    'n_Classes' : 4, #类别数目
    'pre_preparation_save_path': '/mnt/workspace/Baseline/bci42a_data/Standard_input', # 预处理结果保存路径
    'Bandpass_filter_params' : { 'lowcut' : 0.05, 'highcut' : 40, 'fs' : 125, 'order' : 2 }, # 带通滤波器参数
    'Preprocessed_save_path' : "/mnt/workspace/Baseline/bci42a_data/Preprocessed_data",
    'Feature_extracted_save_path' : '/mnt/workspace/Baseline/bci42a_data/Feature_extracted'
}


def down_sample(data, start, end, fs_raw, fs_down):
    '''
    下采样
    
    param---
    data: numpy数组, 一个trial, shape = (n_channels, n_times)
    start: trial有效点起始index
    end: trial有效点结尾index
    
    return---
    data: 下采样之后的数组
    '''
    
    step = fs_raw // fs_down
    data = data[:, start: end: step]
    
    return data

def selected_channel(data, channel_index_list):
    '''
    选择有效的通道
    
    param---
    data: numpy数组, 一个trial, shape = (n_channels, n_times)
    channel_index_list: list类型, 有效通道的index
    '''
    return data[channel_index_list, :]



def pre_preparation(data_dict, config):
    '''
    预处理: 下采样+选择通道, 并保存数据(.npy文件，文件名为标签)
    
    param---
    data_dict: 字典类型, key为一个标识，value为数据(numpy类型, shape = (n, n_times, n_channels))
    n_Class: 类别数目
    
    return---
    '''
    data = [[] for _ in range(config['n_Classes'])]
            
    
    if not os.path.exists(config['pre_preparation_save_path']):
        os.makedirs(config['pre_preparation_save_path'])

    for key, value in data_dict.items():
        d = data[config['file_label_dict'][key]]
        for trial in value:
            # print(np.array(trial).shape)
            trial = down_sample(np.array(trial).T, config['start'], config['end'], config['fs_raw'], config['fs_down'])
            trail = selected_channel(trial, config['channel_index_list'])
            d.append(trail)
    
    # save
    for i, d in enumerate(data):
        data[i] = np.array(d)
        path = config['pre_preparation_save_path'] + "/" + str(i) + ".npy"
        np.save(path, data[i])

    return data


def pre_preparation_bcic(data_path, config):
    '''
    预处理: 下采样+选择通道, 并保存数据(.npy文件，文件名和.mat一样)
    
    param---
    data_path: list类型, ex. ['s001.mat', 'se001.mat']
    
    return---
    '''

    trials = []
    trials_label = []
    for path in data_path:
        d = loadmat(path)
        ntrials = d['x'].shape[2]
        trials_label.append(np.array(d['y']).reshape(-1))
        t = []
        for i in range(ntrials):
            x = down_sample(d['x'][:,:,i], config['start'], config['end'], config['fs_raw'], config['fs_down'])
            x = selected_channel(x, config['channel_index_list'])
            t.append(x)
        trials.append(np.array(t))

    if not os.path.exists(config['pre_preparation_save_path']):
        os.makedirs(config['pre_preparation_save_path'])

    for i, path in enumerate(data_path):
        path = config['pre_preparation_save_path'] + '/' + path[-8:-4] + '.npy'
        labels_path = config['pre_preparation_save_path'] + '/' + path[-8:-4] + '_lables.npy'
        np.save(path, trials[i])
        np.save(labels_path, trials_label[i])

    return trials, trials_label

def butter_bandpass_filter(data, config):
    fa = 0.5 * config['fs']
    low = config['lowcut'] / fa
    high = config['highcut'] / fa
    b, a = butter(config['order'], [low, high], btype='band')
    ret = []
    for line in data:
        ret.append(filtfilt(b, a, line))
    return np.array(ret)

def iirnotch_filter(data, fs = 125, Q = 30, f_cut = 50.0):
    ret = []
    b, a = iirnotch(f_cut, Q, fs)
    for line in data:
        ret.append(lfilter(b,a, line))
    return np.array(ret)

save_path = os.getcwd() + "/lyh_data/session1/Preprocessed_data"


def pre_processing(data, config, butter_order = 2):
    
    save_path = config['Preprocessed_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for data_per_class in data:
        for i, d in enumerate(data_per_class):
            assert d.shape == (14, 500)
            # data_per_class[i] = baseline_correction(data_per_class[i], baseline_start, baseline_end)
            data_per_class[i] = butter_bandpass_filter(data_per_class[i], config['Bandpass_filter_params'])
            data_per_class[i] = iirnotch_filter(data_per_class[i])
    
    # save
    for i, d in enumerate(data):
        data[i] = np.array(d)
        path = save_path + "/" + str(i) + ".npy"
        np.save(path, data[i])
    
    return data

def pre_processing_bcic(data_path, labels_path, config, butter_order = 2):
    
    save_path = config['Preprocessed_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for i in range(len(data_path)):
        trials, labels = np.load(data_path[i]), np.load(labels_path[i])
        for j in range(trials.shape[0]):
            assert trials[j].shape == (22, 500)
            trials[j] = butter_bandpass_filter(trials[j], config['Bandpass_filter_params'])
            trials[j] = iirnotch_filter(trials[j])

        # save
        path1 = config['Preprocessed_save_path'] + '/' + data_path[i][-8:-4] + '.npy'
        path2 = config['Preprocessed_save_path'] + '/' + data_path[i][-8:-4] + '_lables.npy'
        print(trials.shape, labels.shape)
        np.save(path1, trials)
        np.save(path2, labels)
        


# 计算过零率
def zero_crossing_rate(trials):
    '''
    计算一个trials(二维列表)各个通道的过零率, 返回一个list
    '''

    def compute(signal):
        '''
        计算一维信号的过零率
        '''
        crossings = np.where(np.diff(np.sign(signal)))[0]
        zero_crossing_rate = len(crossings) / len(signal)
        return zero_crossing_rate

    ret = [compute(trials[i]) for i in range(len(trials))]
    return np.array(ret)

# 计算均值

def calculate_channel_means(eeg_data):
    """
    计算每个通道的均值。

    参数：
    - eeg_data: 二维 NumPy 数组，表示 EEG 数据，形状为 (n_channels, n_points)。

    返回：
    - channel_means: 一维 NumPy 数组，包含每个通道的均值。
    """
    # 计算每个通道的均值，axis=2 表示在样本点上求均值
    channel_means = np.mean(eeg_data, axis=1)

    return channel_means

# 计算标准差

def calculate_channel_stds(eeg_data):
    """
    计算每个通道的均值。

    参数：
    - eeg_data: 二维 NumPy 数组，表示 EEG 数据，形状为 (n_channels, n_points)。

    返回：
    - channel_means: 一维 NumPy 数组，包含每个通道的均值。
    """
    # 计算每个通道的均值，axis=2 表示在样本点上求均值
    channel_stds = np.std(eeg_data, axis=1)

    return channel_stds

def calculate_first_order_diff(eeg_data):

    first_order_diff = np.sum(np.abs(np.diff(eeg_data, axis=1)), axis=1) / (eeg_data.shape[1] - 1)

    return first_order_diff

def calculate_second_order_diff(eeg_data):
    
    second_order_diff = np.sum(np.abs(np.diff(eeg_data, n=2, axis=1)), axis=1) / (eeg_data.shape[1] - 2)

    return second_order_diff

def time_domain_feature(eeg_dataset):
    '''
    提取时域特征
    params:
    - eeg_dataset: Dataframe类型, 必须包含两列，一列是'label'， 一列是'raw data'
    '''
    zero_crossing_list = []
    means_list = []
    stds_list = []
    first_order_diff_list = []
    second_order_diff_list = []

    for _, row in eeg_dataset.iterrows():
        zero_crossing = zero_crossing_rate(row['raw data'])
        zero_crossing_list.append(zero_crossing)
        means = calculate_channel_means(row['raw data'])
        means_list.append(means)
        stds = calculate_channel_stds(row['raw data'])
        stds_list.append(stds)
        first_diff = calculate_first_order_diff(row['raw data'])
        first_order_diff_list.append(first_diff)
        second_diff = calculate_second_order_diff(row['raw data'])
        second_order_diff_list.append(second_diff)
    
    nChannels = zero_crossing_list[0].shape[0]
    for i in range(nChannels):
        eeg_dataset[f'zero_crossing_rate_c{i+1}'] = np.array(zero_crossing_list)[:,i]
        eeg_dataset[f'mean_c{i+1}'] = np.array(means_list)[:,i]
        eeg_dataset[f'std_c{i+1}'] = np.array(stds_list)[:,i]
        eeg_dataset[f'first_order_diff_c{i+1}'] = np.array(first_order_diff_list)[:,i]
        eeg_dataset[f'second_order_diff_c{i+1}'] = np.array(second_order_diff_list)[:,i]

    return eeg_dataset

def five_band_energy(eeg_data, fs=125):
    
    energy = []
    
    for eeg_signal in eeg_data:
        # 计算功率谱密度（PSD）
        frequencies, psd = welch(eeg_signal, fs, nperseg=1024)

        # 定义频带边界
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 14)
        beta_band = (14, 30)
        gamma_band = (30, 60)

        # 计算每个频带内的能量
        energy.append(np.trapz(psd[(frequencies >= delta_band[0]) & (frequencies <= delta_band[1])], \
            frequencies[(frequencies >= delta_band[0]) & (frequencies <= delta_band[1])]))
        energy.append(np.trapz(psd[(frequencies >= theta_band[0]) & (frequencies <= theta_band[1])], \
            frequencies[(frequencies >= theta_band[0]) & (frequencies <= theta_band[1])]))
        energy.append(np.trapz(psd[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])], \
            frequencies[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])]))
        energy.append(np.trapz(psd[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1])], \
            frequencies[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1])]))
        energy.append(np.trapz(psd[(frequencies >= gamma_band[0]) & (frequencies <= gamma_band[1])], \
            frequencies[(frequencies >= gamma_band[0]) & (frequencies <= gamma_band[1])]))

    return np.array(energy)

def freq_domain_feature(eeg_dataset):
    '''
    提取时域特征

    参数:
    - eeg_dataset: Dataframe类型
    '''
    
    energy = []

    for _, row in eeg_dataset.iterrows():
        
        energy.append(five_band_energy(row['raw data']))

    freq_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    nFreqs = len(freq_names)
    nChannels = energy[0].shape[0] // nFreqs
    
    for i in range(nChannels):
        for j in range(nFreqs):
            eeg_dataset[f'{freq_names[j]}_c{i+1}'] = np.array(energy)[:,i*nFreqs+j]

    return eeg_dataset

def feature_extracting(data_df, config):
    data_df = time_domain_feature(data_df)
    data_df = freq_domain_feature(data_df)

    save_path = config['Feature_extracted_save_path']
    print(config)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path += "/feature_extracted.csv"
    data_df.to_csv(save_path, index=False)
    

def feature_extracting_bcic(data_df, save_file_name, config):
    data_df = time_domain_feature(data_df)
    data_df = freq_domain_feature(data_df)

    save_path = config['Feature_extracted_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path = save_path + "/" + save_file_name
    
    data_df.to_csv(save_path, index=False)

    print(data_df.info())



if __name__ == '__main__':
    # raw_data_dict = read_folder_npy_data()
    # standard_input_data_list = pre_preparation(raw_data_dict, config)
    # preprocessed_data = pre_processing(standard_input_data_list, config)
    # dataset_np = np.array(preprocessed_data)
    # dataset_np = dataset_np.reshape(-1, 14, 500)
    # label_list = [0] * 300 + [1] * 300 + [2] * 300 + [3] * 300
    # data_df = pd.DataFrame({'raw data': list(dataset_np), 'label': label_list})
    # feature_extracting(data_df, config)


    # a = ['/mnt/workspace/Baseline/bci42a_data/Standard_input/s004.npy', \
    #     '/mnt/workspace/Baseline/bci42a_data/Standard_input/e004.npy']
    # b = ['/mnt/workspace/Baseline/bci42a_data/Standard_input/s004_lables.npy', \
    #     '/mnt/workspace/Baseline/bci42a_data/Standard_input/e004_lables.npy']
    # pre_processing_bcic(a, b, config)
    trials = np.load('/mnt/workspace/Baseline/bci42a_data/Preprocessed_data/e001.npy')
    labels = np.load('/mnt/workspace/Baseline/bci42a_data/Preprocessed_data/e001_lables.npy')
    data_df = pd.DataFrame({'raw data': list(trials), 'label': labels})
    feature_extracting_bcic(data_df, 'e001.csv', config)
