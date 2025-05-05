import os
import numpy as np
import mne
import argparse
from scipy.signal import resample
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

# 常量定义
SAMPLE_RATE = 128  # 采样率
SAMPLE_LEN = 128   # 一个样本的时间戳数量
TARGET_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                   'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
# 某些数据集可能使用不同的命名规则，需要进行映射
CHANNEL_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'
}

def parse_args():
    parser = argparse.ArgumentParser(description='预处理EEG数据为LEAD模型可用格式')
    parser.add_argument('--input_dir', type=str, required=True, help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出数据目录')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--original_fs', type=int, default=None, help='原始数据采样率')
    parser.add_argument('--ad_class', type=int, default=1, help='AD标签值，默认为1')
    parser.add_argument('--normal_class', type=int, default=0, help='正常标签值，默认为0')
    return parser.parse_args()

def create_directory(path):
    """创建目录，如果不存在的话"""
    if not os.path.exists(path):
        os.makedirs(path)

def align_channels(raw, target_channels=TARGET_CHANNELS):
    """将通道对齐到目标19通道"""
    # 标准化通道名称 (处理可能的别名，如T7->T3, T8->T4等)
    ch_names = raw.info['ch_names']
    standardized_ch_names = []
    for ch in ch_names:
        if ch in CHANNEL_ALIASES:
            standardized_ch_names.append(CHANNEL_ALIASES[ch])
        else:
            standardized_ch_names.append(ch)
    
    # 更新通道名称
    for i, ch in enumerate(ch_names):
        if ch in CHANNEL_ALIASES:
            raw.rename_channels({ch: CHANNEL_ALIASES[ch]})
    
    # 检查缺失通道
    missing_channels = [ch for ch in target_channels if ch not in raw.info['ch_names']]
    if missing_channels:
        print(f"需要插值的通道: {missing_channels}")
        
        # 如果缺少3个以上通道，尝试直接使用标准10-20系统的蒙太奇
        if len(missing_channels) > 3:
            try:
                # 添加标准10-20系统电极位置信息
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage)
                
                # 插值缺失的通道
                raw = raw.copy().add_reference_channels(missing_channels)
                raw = raw.interpolate_bads(reset_bads=True)
            except Exception as e:
                print(f"自动插值失败: {e}")
                print("尝试手动添加缺失通道并插值")
                # 手动添加缺失通道
                for ch in missing_channels:
                    raw = raw.copy().add_channels([mne.io.RawArray(
                        np.zeros((1, len(raw.times))), 
                        mne.create_info([ch], raw.info['sfreq'], ['eeg'])
                    )])
                
                # 设置这些通道为坏道并插值
                raw.info['bads'] = missing_channels
                raw = raw.interpolate_bads(reset_bads=True)
    
    # 选择所需的通道
    raw = raw.pick_channels(target_channels, ordered=True)
    
    return raw

def resample_time_series(data, original_fs, target_fs):
    """将时间序列数据从原始采样率重采样到目标采样率"""
    T, C = data.shape
    new_length = int(T * target_fs / original_fs)

    resampled_data = np.zeros((new_length, C))
    for i in range(C):
        resampled_data[:, i] = resample(data[:, i], new_length)

    return resampled_data

def split_eeg_segments(data, segment_length=SAMPLE_LEN):
    """将EEG数据分割为固定长度的片段"""
    T, C = data.shape
    num_segments = T // segment_length
    reshaped_data = data[:num_segments * segment_length].reshape(num_segments, segment_length, C)

    return reshaped_data

def process_eeg_file(file_path, original_fs):
    """处理单个EEG文件"""
    # 根据文件扩展名选择适当的读取方法
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.edf':
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif ext == '.set':
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif ext in ['.cnt', '.vhdr']:
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    # 如果未提供原始采样率，从数据中获取
    if original_fs is None:
        original_fs = raw.info['sfreq']
    
    # 通道对齐
    raw = align_channels(raw)
    
    # 获取EEG数据
    data = raw.get_data().T  # (timepoints, channels)
    
    return data, original_fs

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    feature_dir = os.path.join(output_dir, 'Feature')
    label_dir = os.path.join(output_dir, 'Label')
    create_directory(feature_dir)
    create_directory(label_dir)
    
    # 扫描输入目录
    subject_ids = []
    subject_labels = []
    subject_counter = 1
    
    # 假设目录结构: input_dir/[AD或Normal]/subject_files
    for group in os.listdir(args.input_dir):
        group_path = os.path.join(args.input_dir, group)
        if not os.path.isdir(group_path):
            continue
        
        # 根据文件夹名称判断类别
        if 'ad' in group.lower() or 'alzheimer' in group.lower() or 'patient' in group.lower():
            label = args.ad_class
        else:
            label = args.normal_class
        
        # 处理该组的所有受试者
        for subject in os.listdir(group_path):
            subject_path = os.path.join(group_path, subject)
            if not os.path.isdir(subject_path):
                # 如果是单个文件而不是目录，则跳过
                continue
            
            print(f"处理受试者: {subject} (组: {group}, 标签: {label})")
            
            # 收集该受试者的所有EEG文件
            eeg_files = []
            for root, _, files in os.walk(subject_path):
                for file in files:
                    if file.endswith(('.edf', '.set', '.cnt', '.vhdr')):
                        eeg_files.append(os.path.join(root, file))
            
            if not eeg_files:
                print(f"未找到受试者 {subject} 的EEG文件，跳过")
                continue
            
            # 处理该受试者的所有EEG文件并合并
            all_segments = []
            for eeg_file in eeg_files:
                try:
                    print(f"  处理文件: {os.path.basename(eeg_file)}")
                    data, original_fs = process_eeg_file(eeg_file, args.original_fs)
                    
                    # 重采样
                    resampled_data = resample_time_series(data, original_fs, SAMPLE_RATE)
                    
                    # 分段
                    segments = split_eeg_segments(resampled_data, SAMPLE_LEN)
                    
                    if len(segments) > 0:
                        all_segments.append(segments)
                    else:
                        print(f"  警告: 文件 {os.path.basename(eeg_file)} 太短，无法分段")
                except Exception as e:
                    print(f"  处理文件 {os.path.basename(eeg_file)} 时出错: {e}")
            
            if all_segments:
                # 合并所有分段
                all_segments = np.vstack(all_segments)
                print(f"  最终形状: {all_segments.shape}")
                
                # 保存特征
                np.save(os.path.join(feature_dir, f'feature_{subject_counter:02d}.npy'), all_segments)
                
                # 记录受试者ID和标签
                subject_ids.append(subject_counter)
                subject_labels.append(label)
                
                subject_counter += 1
            else:
                print(f"  警告: 受试者 {subject} 没有有效的EEG片段，跳过")
    
    # 创建并保存标签文件
    labels = np.column_stack((subject_labels, subject_ids))
    np.save(os.path.join(label_dir, 'label.npy'), labels)
    
    print(f"\n处理完成! 处理了 {len(subject_ids)} 个受试者，数据保存到 {output_dir}")
    print(f"AD受试者: {labels[labels[:, 0] == args.ad_class].shape[0]}")
    print(f"正常受试者: {labels[labels[:, 0] == args.normal_class].shape[0]}")

if __name__ == "__main__":
    main() 