import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
)
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.signal import resample

from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
from data_provider.dataset_loader.adfsu_loader import ADFSULoader
from data_provider.dataset_loader.cognision_rseeg_loader import COGrsEEGLoader
from data_provider.dataset_loader.cognision_erp_loader import COGERPLoader
from data_provider.dataset_loader.adftd_loader import ADFTDLoader
from data_provider.dataset_loader.cnbpm_loader import CNBPMLoader
from data_provider.dataset_loader.brainlat_loader import BrainLatLoader
from data_provider.dataset_loader.ad_auditory_loader import ADAuditoryLoader
from data_provider.dataset_loader.tdbrain_loader import TBDRAINLoader
from data_provider.dataset_loader.tuep_loader import TUEPLoader
from data_provider.dataset_loader.reeg_pd_loader import REEGPDLoader
from data_provider.dataset_loader.pearl_neuro_loader import PEARLNeuroLoader
from data_provider.dataset_loader.depression_loader import DepressionLoader
from data_provider.dataset_loader.reeg_srm_loader import REEGSRMLoader
from data_provider.dataset_loader.reeg_baca_loader import REEGBACALoader

# 数据集名称到数据加载器的映射字典
data_folder_dict = {
    # 注意事项：
    # 1. 键名必须与数据集文件夹名称相同
    # 2. 原始通道数为19的数据集不需要添加"-19"后缀
    
    # 原始通道数的数据集（用于单数据集监督学习）
    'APAVA': APAVALoader,  # APAVA数据集，16通道EEG
    'Cognision-ERP': COGERPLoader,  # Cognision事件相关电位数据集，7通道EEG
    'Cognision-rsEEG': COGrsEEGLoader,  # Cognision静息态EEG数据集，7通道EEG
    'BrainLat': BrainLatLoader,  # BrainLat数据集，128通道EEG

    # 统一使用19通道配置的数据集（保证通道一致性）
    # 5个下游任务数据集
    'ADFTD': ADFTDLoader,  # 阿尔茨海默病功能连接数据集，19通道
    'CNBPM': CNBPMLoader,  # 认知障碍生物标志物数据集，19通道
    'Cognision-ERP-19': COGERPLoader,  # Cognision ERP的19通道版本
    'Cognision-rsEEG-19': COGrsEEGLoader,  # Cognision rsEEG的19通道版本
    'BrainLat-19': BrainLatLoader,  # BrainLat的19通道版本

    # 11个预训练数据集
    'ADSZ': ADSZLoader,  # 癫痫失神发作数据集，19通道
    'ADFSU': ADFSULoader,  # 阿尔茨海默病功能研究数据集，19通道
    'APAVA-19': APAVALoader,  # APAVA的19通道版本
    'AD-Auditory': ADAuditoryLoader,  # 阿尔茨海默病听觉刺激数据集
    'TDBRAIN-19': TBDRAINLoader,  # 典型发育脑数据集，19通道
    'TUEP': TUEPLoader,  # 天普大学癫痫数据集，19通道
    'REEG-PD-19': REEGPDLoader,  # 帕金森病静息态EEG数据集，19通道
    'PEARL-Neuro-19': PEARLNeuroLoader,  # PEARL神经科学研究数据集，19通道
    'Depression-19': DepressionLoader,  # 抑郁症EEG数据集，19通道
    'REEG-SRM-19': REEGSRMLoader,  # 感觉运动节律EEG数据集，19通道
    'REEG-BACA-19': REEGBACALoader,  # 脑机接口应用EEG数据集，19通道
}

# 忽略警告信息
warnings.filterwarnings('ignore')


class SingleDatasetLoader(Dataset):
    """单数据集加载器，继承自PyTorch的Dataset类"""
    
    def __init__(self, args, root_path, flag=None):
        """初始化单数据集加载器
        
        Args:
            args: 包含所有配置参数的对象
            root_path: 数据集根目录路径
            flag: 数据标志（PRETRAIN/TRAIN/TEST/VAL）
        """
        self.no_normalize = args.no_normalize  # 是否禁用数据标准化
        self.root_path = root_path  # 数据集根路径

        print(f"Loading {flag} samples from single dataset...")
        # 根据flag类型获取对应的数据集列表
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")  # 预训练数据集
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")  # 训练数据集
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")  # 测试/验证数据集
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
            
        # 验证只传入了一个数据集
        if len(data_folder_list) > 1:
            raise ValueError("Only one dataset should be given here")
            
        print(f"Datasets used ", data_folder_list[0])
        data = data_folder_list[0]  # 获取数据集名称
        
        # 检查数据集是否在预定义的字典中
        if data not in data_folder_dict.keys():
            raise Exception("Data not matched, please check if the data folder name in data_folder_dict.")
        else:
            # 获取对应的数据集加载器类
            Data = data_folder_dict[data]
            # 实例化数据集加载器
            data_set = Data(
                root_path=os.path.join(args.root_path, data),  # 完整数据集路径
                args=args,  # 配置参数
                flag=flag,  # 数据标志
            )
            print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
            
            # 为标签添加数据集ID（单数据集固定为1）
            data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, 1).reshape(-1, 1)), axis=1)
            self.X, self.y = data_set.X, data_set.y  # 保存数据和标签

        # 打乱数据顺序（固定随机种子42保证可重复性）
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]  # 记录最大序列长度
        print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        """获取单个样本
        
        Args:
            index: 样本索引
        Returns:
            包含数据和标签的元组（已转换为PyTorch张量）
        """
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        """返回数据集大小"""
        return len(self.y)


class MultiDatasetsLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from multiple datasets...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        print(f"Datasets used ", data_folder_list)
        self.X, self.y = None, None
        global_sub_num = 1  # count global subject number to avoid duplicate IDs in multiple datasets
        for i, data in enumerate(data_folder_list):
            if data not in data_folder_dict.keys():
                raise Exception("Data not matched, "
                                "please check if the data folder name in data_folder_dict.")
            else:
                Data = data_folder_dict[data]
                data_set = Data(
                    root_path=os.path.join(args.root_path, data),
                    args=args,
                    flag=flag,
                )
                # add dataset ID to the third column of y, id starts from 1
                data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, i + 1).reshape(-1, 1)), axis=1)
                print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
                if self.X is None or self.y is None:
                    self.X, self.y = data_set.X, data_set.y
                    global_sub_num = len(data_set.all_ids)
                else:
                    # number of subjects in the current dataset
                    local_sub_num = len(data_set.all_ids)
                    # update subject IDs in the current dataset by adding global_sub_num
                    data_set.y[:, 1] += global_sub_num
                    # update global subject number
                    global_sub_num += local_sub_num
                    # concatenate data from different datasets
                    self.X, self.y = (np.concatenate((self.X, data_set.X), axis=0),
                                      np.concatenate((self.y, data_set.y), axis=0))

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        # print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)

