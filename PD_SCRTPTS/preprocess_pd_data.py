import os
import numpy as np
import scipy.io
import mne
import argparse
from scipy.signal import resample
from sklearn.utils import shuffle
import warnings
import pandas as pd
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
    parser = argparse.ArgumentParser(description='预处理PD数据为LEAD模型可用格式')
    parser.add_argument('--input_dir', type=str, default='../PD_DATASET', 
                        help='输入数据目录，包含PD和Normal子目录')
    parser.add_argument('--output_dir', type=str, default='../dataset', 
                        help='输出数据目录')
    parser.add_argument('--dataset_name', type=str, default='PD_REST', help='数据集名称')
    parser.add_argument('--original_fs', type=int, default=500, help='原始数据采样率，默认500Hz')
    parser.add_argument('--pd_class', type=int, default=2, help='PD患者标签值，默认为2')
    parser.add_argument('--normal_class', type=int, default=0, help='正常人标签值，默认为0')
    parser.add_argument('--metadata_file', type=str, default=None, 
                        help='可选的元数据文件路径，例如Excel文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    return parser.parse_args()

def create_directory(path):
    """创建目录，如果不存在的话"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_mat_file(file_path, original_fs, verbose=False):
    """加载.mat文件并转换为MNE Raw对象
    
    针对EEGLAB格式的传统MATLAB文件(v7.2或更早)进行优化处理
    """
    try:
        if verbose:
            print(f"开始加载文件: {file_path}")
            
        # 加载.mat文件
        try:
            mat_data = scipy.io.loadmat(file_path)
            if verbose:
                print("成功以传统MATLAB格式加载文件")
        except Exception as e:
            print(f"尝试以传统格式加载失败: {e}")
            print("尝试使用h5py加载HDF5格式...")
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    # 只负责诊断，实际加载还是使用MNE的功能
                    if verbose:
                        print("文件为HDF5格式，包含以下键:")
                        for key in f.keys():
                            print(f"  - {key}")
                    pass
            except Exception as h5_e:
                print(f"使用h5py加载也失败: {h5_e}")
            
            # 尝试使用MNE的EEGLAB读取器
            try:
                print("尝试使用MNE的EEGLAB读取器...")
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
                print("成功使用MNE的EEGLAB读取器加载文件")
                return raw
            except Exception as mne_e:
                print(f"MNE的EEGLAB读取也失败: {mne_e}")
                raise ValueError(f"无法加载文件 {file_path}，请确保安装了h5py库或检查文件格式")
        
        # 处理EEGLAB格式数据
        if 'EEG' in mat_data:
            if verbose:
                print("发现EEGLAB格式数据")
                
            # 获取EEG结构体的第一个元素
            eeg_struct = mat_data['EEG'][0, 0]
            
            # 诊断打印 - 显示所有可用的字段
            if verbose:
                print("EEGLAB结构体包含以下字段:")
                for field in eeg_struct.dtype.names:
                    field_value = eeg_struct[field]
                    if isinstance(field_value, np.ndarray):
                        print(f"  - {field}: 形状={field_value.shape}, 类型={field_value.dtype}")
                    else:
                        print(f"  - {field}: {type(field_value)}")
            
            # 获取EEG数据 - 直接访问已知的结构
            if 'data' in eeg_struct.dtype.names:
                eeg_data = eeg_struct['data']
                if verbose:
                    print(f"EEG数据形状: {eeg_data.shape}")
            else:
                raise ValueError("在EEGLAB结构中找不到'data'字段")
            
            # 获取采样率
            if 'srate' in eeg_struct.dtype.names:
                fs = float(eeg_struct['srate'][0, 0])
                if verbose:
                    print(f"采样率: {fs} Hz")
            else:
                fs = original_fs
                if verbose:
                    print(f"使用默认采样率: {fs} Hz")
            
            # 获取通道信息
            channel_names = []
            if 'chanlocs' in eeg_struct.dtype.names:
                chanlocs = eeg_struct['chanlocs']
                if verbose:
                    print(f"发现通道位置信息: {chanlocs.shape[1]}个通道")
                    
                    # 诊断第一个通道的结构
                    if chanlocs.shape[1] > 0:
                        first_chan = chanlocs[0, 0]
                        print("第一个通道的字段:")
                        for chan_field in first_chan.dtype.names:
                            print(f"  - {chan_field}")
                
                # 从chanlocs中提取通道名称
                for i in range(chanlocs.shape[1]):
                    chan = chanlocs[0, i]
                    if 'labels' in chan.dtype.names:
                        # 获取标签值
                        label = chan['labels']
                        if isinstance(label, np.ndarray) and label.size > 0:
                            if isinstance(label[0], np.ndarray):
                                channel_names.append(str(label[0][0]))
                            else:
                                channel_names.append(str(label[0]))
                        else:
                            channel_names.append(f'Ch{i+1}')
                    else:
                        channel_names.append(f'Ch{i+1}')
            else:
                # 如果找不到通道名称，生成默认名称
                channel_names = [f'Ch{i+1}' for i in range(eeg_data.shape[0])]
                
            if verbose:
                print(f"通道名称: {channel_names[:5]}... (共{len(channel_names)}个)")
                
            # 创建MNE Raw对象
            info = mne.create_info(channel_names, sfreq=fs, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info)
            
            return raw
            
        else:
            # 如果不是EEGLAB格式，尝试查找数据数组
            potential_data_vars = ['data', 'eeg', 'EEG', 'Data', 'Signal', 'signal']
            
            # 诊断打印 - 显示所有顶级变量
            if verbose:
                print("文件中包含以下顶级变量:")
                for var_name in mat_data.keys():
                    if not var_name.startswith('__'):  # 排除内部变量
                        var_value = mat_data[var_name]
                        if isinstance(var_value, np.ndarray):
                            print(f"  - {var_name}: 形状={var_value.shape}, 类型={var_value.dtype}")
                        else:
                            print(f"  - {var_name}: {type(var_value)}")
            
            # 找到数据数组
            data_var = None
            for var in potential_data_vars:
                if var in mat_data:
                    data_var = var
                    break
                    
            if data_var is None:
                # 如果没有找到预定义的变量名，尝试查找最大的数组
                largest_var = None
                largest_size = 0
                for var_name, var_data in mat_data.items():
                    if isinstance(var_data, np.ndarray) and not var_name.startswith('__'):
                        if var_data.size > largest_size:
                            largest_size = var_data.size
                            largest_var = var_name
                
                if largest_var:
                    data_var = largest_var
                    if verbose:
                        print(f"选择最大的数组变量: {data_var}，形状={mat_data[data_var].shape}")
                    
            if data_var is None:
                raise ValueError(f"无法在{file_path}中找到EEG数据数组")
                
            # 获取EEG数据
            eeg_data = mat_data[data_var]
            
            # 检查并适应数据形状 - 假设数据格式可能是(channels, samples)或(samples, channels)
            if len(eeg_data.shape) == 2:
                # 如果通道数少于时间点，假设格式是(channels, samples)
                if eeg_data.shape[0] <= 128 and eeg_data.shape[1] > 1000:  # 通常EEG通道不超过128个，且采样点很多
                    # 已经是(channels, samples)格式
                    if verbose:
                        print(f"数据形状符合(channels, samples)格式: {eeg_data.shape}")
                else:
                    # 假设是(samples, channels)，需要转置
                    if verbose:
                        print(f"数据形状可能是(samples, channels)，进行转置，原始形状: {eeg_data.shape}")
                    eeg_data = eeg_data.T
                    if verbose:
                        print(f"转置后形状: {eeg_data.shape}")
                    
            # 生成默认通道名称
            channel_names = [f'Ch{i+1}' for i in range(eeg_data.shape[0])]
            
            # 创建MNE原始对象
            info = mne.create_info(channel_names, sfreq=original_fs, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info)
            
            return raw
    
    except Exception as e:
        print(f"无法加载文件 {file_path}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise

def align_channels(raw, target_channels=TARGET_CHANNELS, verbose=False):
    """将通道对齐到目标19通道"""
    if verbose:
        print(f"开始对齐通道，原始通道: {len(raw.ch_names)}，目标通道: {len(target_channels)}")
        print(f"原始通道名称: {raw.ch_names[:5]}...")
    
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
        if verbose:
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
                if verbose:
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
    
    if verbose:
        print(f"通道对齐完成，最终通道: {raw.ch_names}")
    
    return raw

def resample_time_series(data, original_fs, target_fs, verbose=False):
    """将时间序列数据从原始采样率重采样到目标采样率"""
    if verbose:
        print(f"重采样: {original_fs}Hz -> {target_fs}Hz")
        
    T, C = data.shape
    new_length = int(T * target_fs / original_fs)

    resampled_data = np.zeros((new_length, C))
    for i in range(C):
        resampled_data[:, i] = resample(data[:, i], new_length)

    if verbose:
        print(f"重采样后形状: {resampled_data.shape}")
        
    return resampled_data

def split_eeg_segments(data, segment_length=SAMPLE_LEN, verbose=False):
    """将EEG数据分割为固定长度的片段"""
    T, C = data.shape
    num_segments = T // segment_length
    
    if verbose:
        print(f"分段: 总长度={T}，段长={segment_length}，分段数={num_segments}")
    
    reshaped_data = data[:num_segments * segment_length].reshape(num_segments, segment_length, C)

    return reshaped_data

def process_eeg_file(file_path, original_fs, verbose=False):
    """处理单个EEG文件"""
    # 根据文件扩展名选择适当的读取方法
    ext = os.path.splitext(file_path)[1].lower()
    
    if verbose:
        print(f"处理文件: {file_path}，格式: {ext}")
    
    if ext == '.mat':
        raw = load_mat_file(file_path, original_fs, verbose)
    elif ext == '.edf':
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
    
    if verbose:
        print(f"原始数据信息: 通道数={len(raw.ch_names)}, 采样率={raw.info['sfreq']}Hz, 时长={len(raw.times)/raw.info['sfreq']:.2f}秒")
    
    # 通道对齐
    raw = align_channels(raw, verbose=verbose)
    
    # 获取EEG数据
    data = raw.get_data().T  # (timepoints, channels)
    
    if verbose:
        print(f"提取数据形状: {data.shape}")
    
    return data, original_fs

def load_metadata(metadata_file, verbose=False):
    """加载元数据文件，如Excel"""
    if metadata_file and os.path.exists(metadata_file):
        ext = os.path.splitext(metadata_file)[1].lower()
        if ext == '.xlsx' or ext == '.xls':
            if verbose:
                print(f"加载Excel元数据文件: {metadata_file}")
            return pd.read_excel(metadata_file)
        elif ext == '.csv':
            if verbose:
                print(f"加载CSV元数据文件: {metadata_file}")
            return pd.read_csv(metadata_file)
    return None

def main():
    args = parse_args()
    verbose = args.verbose
    
    if verbose:
        print(f"开始处理PD数据集，输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}/{args.dataset_name}")
        print(f"原始采样率: {args.original_fs}Hz")
        print(f"标签设置: PD={args.pd_class}, Normal={args.normal_class}")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    feature_dir = os.path.join(output_dir, 'Feature')
    label_dir = os.path.join(output_dir, 'Label')
    create_directory(feature_dir)
    create_directory(label_dir)
    
    # 添加一个错误日志文件
    error_log_path = os.path.join(output_dir, 'processing_errors.log')
    with open(error_log_path, 'w', encoding='utf-8') as error_log:
        error_log.write(f"PD数据处理错误日志\n时间: {pd.Timestamp.now()}\n\n")
    
    # 加载元数据
    metadata = None
    if args.metadata_file:
        metadata = load_metadata(args.metadata_file, verbose)
        if metadata is not None and verbose:
            print(f"成功加载元数据: {args.metadata_file}")
            print(f"元数据列: {metadata.columns.tolist()}")
    
    # 扫描PD和Normal目录
    subject_ids = []
    subject_labels = []
    subject_counter = 1
    
    # 处理PD目录
    pd_dir = os.path.join(args.input_dir, 'PD')
    if os.path.exists(pd_dir):
        print(f"处理PD目录: {pd_dir}")
        
        # 遍历受试者目录
        for subject in os.listdir(pd_dir):
            subject_path = os.path.join(pd_dir, subject)
            if not os.path.isdir(subject_path):
                continue
                
            print(f"\n处理帕金森病患者: {subject}")
            
            # 收集该受试者的所有mat文件
            eeg_files = []
            for root, _, files in os.walk(subject_path):
                for file in files:
                    if file.endswith('.mat'):
                        eeg_files.append(os.path.join(root, file))
            
            if not eeg_files:
                print(f"未找到受试者 {subject} 的EEG文件，跳过")
                continue
            
            # 处理该受试者的所有EEG文件并合并
            all_segments = []
            for eeg_file in eeg_files:
                try:
                    print(f"  处理文件: {os.path.basename(eeg_file)}")
                    data, original_fs = process_eeg_file(eeg_file, args.original_fs, verbose)
                    
                    # 重采样
                    resampled_data = resample_time_series(data, original_fs, SAMPLE_RATE, verbose)
                    
                    # 分段
                    segments = split_eeg_segments(resampled_data, SAMPLE_LEN, verbose)
                    
                    if len(segments) > 0:
                        all_segments.append(segments)
                        if verbose:
                            print(f"  成功提取 {len(segments)} 个片段")
                    else:
                        print(f"  警告: 文件 {os.path.basename(eeg_file)} 太短，无法分段")
                except Exception as e:
                    error_msg = f"  处理文件 {os.path.basename(eeg_file)} 时出错: {e}"
                    print(error_msg)
                    # 记录错误到日志
                    with open(error_log_path, 'a', encoding='utf-8') as error_log:
                        error_log.write(f"[{pd.Timestamp.now()}] {error_msg}\n")
                        if verbose:
                            import traceback
                            error_log.write(traceback.format_exc())
                            error_log.write("\n\n")
                    
                    if verbose:
                        import traceback
                        traceback.print_exc()
            
            if all_segments:
                # 合并所有分段
                all_segments = np.vstack(all_segments)
                print(f"  最终形状: {all_segments.shape}")
                
                # 保存特征
                np.save(os.path.join(feature_dir, f'feature_{subject_counter:02d}.npy'), all_segments)
                
                # 记录受试者ID和标签
                subject_ids.append(subject_counter)
                subject_labels.append(args.pd_class)
                
                subject_counter += 1
            else:
                print(f"  警告: 受试者 {subject} 没有有效的EEG片段，跳过")
    
    # 处理Normal目录 - 保持相同逻辑，只需添加相同的错误日志记录逻辑
    normal_dir = os.path.join(args.input_dir, 'Normal')
    if os.path.exists(normal_dir) and os.listdir(normal_dir):  # 确认目录存在且不为空
        print(f"\n处理Normal目录: {normal_dir}")
        
        # 遍历受试者目录
        for subject in os.listdir(normal_dir):
            subject_path = os.path.join(normal_dir, subject)
            if not os.path.isdir(subject_path):
                continue
                
            print(f"\n处理健康对照组: {subject}")
            
            # 收集该受试者的所有mat文件
            eeg_files = []
            for root, _, files in os.walk(subject_path):
                for file in files:
                    if file.endswith('.mat'):
                        eeg_files.append(os.path.join(root, file))
            
            if not eeg_files:
                print(f"未找到受试者 {subject} 的EEG文件，跳过")
                continue
            
            # 处理该受试者的所有EEG文件并合并
            all_segments = []
            for eeg_file in eeg_files:
                try:
                    print(f"  处理文件: {os.path.basename(eeg_file)}")
                    data, original_fs = process_eeg_file(eeg_file, args.original_fs, verbose)
                    
                    # 重采样
                    resampled_data = resample_time_series(data, original_fs, SAMPLE_RATE, verbose)
                    
                    # 分段
                    segments = split_eeg_segments(resampled_data, SAMPLE_LEN, verbose)
                    
                    if len(segments) > 0:
                        all_segments.append(segments)
                        if verbose:
                            print(f"  成功提取 {len(segments)} 个片段")
                    else:
                        print(f"  警告: 文件 {os.path.basename(eeg_file)} 太短，无法分段")
                except Exception as e:
                    error_msg = f"  处理文件 {os.path.basename(eeg_file)} 时出错: {e}"
                    print(error_msg)
                    # 记录错误到日志
                    with open(error_log_path, 'a', encoding='utf-8') as error_log:
                        error_log.write(f"[{pd.Timestamp.now()}] {error_msg}\n")
                        if verbose:
                            import traceback
                            error_log.write(traceback.format_exc())
                            error_log.write("\n\n")
                            
                    if verbose:
                        import traceback
                        traceback.print_exc()
            
            if all_segments:
                # 合并所有分段
                all_segments = np.vstack(all_segments)
                print(f"  最终形状: {all_segments.shape}")
                
                # 保存特征
                np.save(os.path.join(feature_dir, f'feature_{subject_counter:02d}.npy'), all_segments)
                
                # 记录受试者ID和标签
                subject_ids.append(subject_counter)
                subject_labels.append(args.normal_class)
                
                subject_counter += 1
            else:
                print(f"  警告: 受试者 {subject} 没有有效的EEG片段，跳过")
    
    # 创建并保存标签文件
    if subject_ids:
        labels = np.column_stack((subject_labels, subject_ids))
        np.save(os.path.join(label_dir, 'label.npy'), labels)
        
        print(f"\n处理完成! 处理了 {len(subject_ids)} 个受试者，数据保存到 {output_dir}")
        print(f"PD患者: {labels[labels[:, 0] == args.pd_class].shape[0]}")
        print(f"正常对照组: {labels[labels[:, 0] == args.normal_class].shape[0]}")
    else:
        print("\n警告: 未处理任何有效受试者数据!")

if __name__ == "__main__":
    main() 