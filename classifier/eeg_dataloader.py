import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lfilter
from copy import deepcopy
from entropy import *
import entropy
from scipy.signal import welch
from scipy.io import loadmat, savemat


class EEG_DataLoader:
    '''对raw data进行数据预处理, 包括下采样、选择通道、滤波、时域特征提取(过零率、均值、标准差、一阶\二阶差分)、
    频域特征提取(传统五频道法)
    X_raw: raw data, 类型: List[二维numpy arrray, shape=(channels, time_points)]
    y : X_raw 对应的标签
    config: 配置信息
    data_df: 预处理之后的数据, 其中两个colums=['raw data', 'label']不用来训练, 其他的colums都是提取出来的特征,
    用来训练.
    API: self.feature_engineering(): 返回一个datafrmae类型的数据, 可用来训练。
    '''
    def __init__(self, X_raw, y, config):
        self.X_raw = X_raw
        self.y = y
        self.config = config
        self.standard_input_data = []
        self.pre_processed_data = []
        self.data_df = None

    def pre_preparation(self):
        '''
        预处理: 下采样+选择通道
        '''

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

        for trial in self.X_raw:
            trial = down_sample(np.array(trial), self.config['start'], self.config['end'], self.config['fs_raw'], config['fs_down'])
            trial = selected_channel(trial, self.config['channel_index_list'])
            self.standard_input_data.append(trial)

        if self.config['pre_preparation_save_path'] != '':
            if not os.path.exists(self.config['pre_preparation_save_path']):
                os.makedirs(self.config['pre_preparation_save_path'])
            np.save(self.config['pre_preparation_save_path'] + "/standard_input_data" + ".npy", self.standard_input_data)

    def pre_processing(self, butter_order = 2):
        
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

        
        for i in range(len(self.standard_input_data)):
            assert self.standard_input_data[i].shape == (14, 500)
            x = butter_bandpass_filter(self.standard_input_data[i], self.config['Bandpass_filter_params'])
            x = iirnotch_filter(x)
            self.pre_processed_data.append(x)

        if self.config['Preprocessed_save_path'] != '':
            if not os.path.exists(self.config['Preprocessed_save_path']):
                os.makedirs(self.config['Preprocessed_save_path'])
            np.save(self.config['Preprocessed_save_path'] + "/pre_processed_data" + ".npy", self.pre_processed_data)

    def time_domain_feature(self):
        '''
        提取时域特征
        '''
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
        
        zero_crossing_list = []
        means_list = []
        stds_list = []
        first_order_diff_list = []
        second_order_diff_list = []

        for _, row in self.data_df.iterrows():
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
            self.data_df[f'zero_crossing_rate_c{i+1}'] = np.array(zero_crossing_list)[:,i]
            self.data_df[f'mean_c{i+1}'] = np.array(means_list)[:,i]
            self.data_df[f'std_c{i+1}'] = np.array(stds_list)[:,i]
            self.data_df[f'first_order_diff_c{i+1}'] = np.array(first_order_diff_list)[:,i]
            self.data_df[f'second_order_diff_c{i+1}'] = np.array(second_order_diff_list)[:,i]

    def freq_domain_feature(self):
        '''
        提取时域特征

        参数:
        - eeg_dataset: Dataframe类型
        '''

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
        
        energy = []

        for _, row in self.data_df.iterrows():
            
            energy.append(five_band_energy(row['raw data']))

        freq_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        nFreqs = len(freq_names)
        nChannels = energy[0].shape[0] // nFreqs
        
        for i in range(nChannels):
            for j in range(nFreqs):
                self.data_df[f'{freq_names[j]}_c{i+1}'] = np.array(energy)[:,i*nFreqs+j]

    def feature_extracting(self):
        self.data_df = pd.DataFrame({'raw data': self.pre_processed_data, 'label': self.y})
        self.time_domain_feature()
        self.freq_domain_feature()
        if self.config['Feature_extracted_save_path'] != '':
            if not os.path.exists(self.config['Feature_extracted_save_path']):
                os.makedirs(self.config['Feature_extracted_save_path'])
            self.data_df.to_csv(self.config['Feature_extracted_save_path'] + "/feature_extracted_data.csv")

    def feature_engineering(self):
        self.pre_preparation()
        self.pre_processing()
        self.feature_extracting()
        return self.data_df

if __name__ == '__main__':
    filename = '/mnt/workspace/Baseline/Emotiv_dataloader-main/data/nothing1_orginal.npy'
    data = np.load(filename, allow_pickle=True)
    # Print information about the loaded file
    print(f"Loaded {filename}: Shape={data.shape}, Dtype={data.dtype}")
    for i in range(len(data)):
        data[i] = np.array(data[i]).T
    label = [0] * 100
    config = {
        'fs_raw' : 250,  # 原始数据采样频率
        'fs_down' : 125,  # 下采样频率 
        't_of_trial' : 4, # 一个trial的时间, 单位：s
        'start': 2, # trial有效点起始index
        'end': 2+1000, # trial有效点结尾index
        'channel_index_list' : np.arange(2, 2+14),
        'pre_preparation_save_path': '/mnt/workspace/Standard_input', # 预处理结果保存路径
        'Bandpass_filter_params' : { 'lowcut' : 0.05, 'highcut' : 40, 'fs' : 125, 'order' : 2 }, # 带通滤波器参数
        'Preprocessed_save_path' : "/mnt/workspace/Preprocessed_data",
        'Feature_extracted_save_path' : '/mnt/workspace/Feature_extracted'
    }
    data_loader = EEG_DataLoader(data, label, config)
    data_df = data_loader.feature_engineering()
    print(data_df.info())
    print(data_df.columns)

