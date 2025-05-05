# 帕金森病(PD) EEG数据预处理工具使用说明

本文档说明如何使用 `preprocess_pd_data.py` 脚本将帕金森病EEG数据处理为LEAD模型可用的格式。

## 背景

LEAD是一个用于基于EEG的阿尔茨海默病检测的大型基础模型，但它也可以用于其他脑疾病诊断任务，如帕金森病检测。该模型需要特定格式的数据输入：

1. 数据需要重采样至128Hz
2. 每个样本包含128个时间点（1秒）
3. 使用标准的10-20系统的19个通道
4. 数据按受试者组织，包含Feature和Label两个文件夹
5. 数据通过频率过滤和标准化进行预处理

## 安装依赖

在使用脚本前，请先安装必要的依赖：

```bash
pip install numpy scipy mne scikit-learn pandas h5py
```

注意：h5py库是用于支持新版本MATLAB文件(v7.3+，HDF5格式)的读取，强烈建议安装。

## 数据组织

预处理脚本假设您的原始数据按以下结构组织：

```
PD_DATASET/
├── PD/             # 帕金森病患者组
│   ├── 890/        # 受试者890
│   │   ├── 890_1_PD_REST.mat    # EEG记录文件
│   │   └── 890_1_PD_REST1.mat
│   └── 891/        # 受试者891
│       └── ...
└── Normal/         # 健康对照组（如果有）
    ├── subject1/
    │   └── ...
    └── subject2/
        └── ...
```

## 使用方法

基本用法：

```bash
python preprocess_pd_data.py --input_dir ../PD_DATASET --output_dir ../dataset --dataset_name PD_REST
```

启用详细输出（推荐首次运行时使用）：

```bash
python preprocess_pd_data.py --input_dir ../PD_DATASET --output_dir ../dataset --dataset_name PD_REST --verbose
```

### 参数说明

- `--input_dir`：输入数据目录路径，默认为'../PD_DATASET'
- `--output_dir`：处理后数据的输出目录，默认为'../dataset'
- `--dataset_name`：数据集名称，将作为输出目录的子文件夹名，默认为'PD_REST'
- `--original_fs`：原始数据的采样率，默认为500Hz
- `--pd_class`：PD患者标签的值，默认为2
- `--normal_class`：正常人群标签的值，默认为0
- `--metadata_file`：可选的元数据文件路径（如Excel文件），用于额外的临床或人口统计信息
- `--verbose`：启用详细输出，打印处理过程中的更多信息，便于调试和监控

### 示例

使用Excel文件提供额外的元数据信息并启用详细输出：

```bash
python preprocess_pd_data.py --input_dir ../PD_DATASET --output_dir ../dataset --dataset_name PD_REST --metadata_file ../PD_DATASET/IMPORT_ME_REST.xlsx --verbose
```

## 输出格式

处理后的数据将保存在以下结构中：

```
output_dir/dataset_name/
├── Feature/
│   ├── feature_01.npy   # 受试者1的特征数据，形状为(N, 128, 19)
│   ├── feature_02.npy   # 受试者2的特征数据 
│   └── ...              # 更多受试者
├── Label/
│   └── label.npy        # 所有受试者的标签，形状为(M, 2)，每行包含[标签, ID]
└── processing_errors.log  # 处理过程中的错误记录
```

其中：
- `N` 是该受试者的样本数量
- `128` 是每个样本的时间点数量
- `19` 是通道数量
- `M` 是受试者总数
- 标签中，2通常表示PD患者，0表示健康对照组（可通过参数修改）

## 脚本功能详解

此脚本提供了全面的PD数据处理能力：

1. **多格式MATLAB文件支持**：
   - 支持传统MATLAB格式(v7.2或更早)
   - 支持HDF5格式MATLAB文件(v7.3及更高版本)
   - 支持EEGLAB数据结构自动识别

2. **智能数据识别与转换**：
   - 自动识别EEGLAB结构中的关键字段(data, chanlocs, srate等)
   - 能够处理不同数据格式，自动检测和转换数据形状
   - 通道名称提取与映射，支持各种命名约定

3. **高级通道处理**：
   - 标准10-20系统通道对齐
   - 缺失通道智能插值
   - 通道别名支持(T7→T3, T8→T4等)

4. **错误处理与诊断**：
   - 详细的错误日志记录
   - 启用verbose模式时提供丰富的诊断信息
   - 多级错误恢复机制，提高处理可靠性

5. **元数据集成**：
   - 支持Excel或CSV格式的外部元数据
   - 批量处理多受试者数据
   - 灵活的标签分配

## 后续步骤

预处理完成后，您可以将数据放入LEAD模型的`dataset/`目录下对应的数据集文件夹中，然后按照LEAD模型的说明进行训练或推理。与阿尔茨海默病数据一样，您需要创建一个自定义的数据加载器：

1. 创建文件`data_provider/dataset_loader/pd_rest_loader.py`
2. 参考其他数据集的加载器实现自定义加载器
3. 在`data_provider/data_loader.py`中引入并注册您的加载器
4. 使用`run.py`脚本指定您的数据集进行训练或测试

## 故障排除

1. **MATLAB文件读取错误**：
   - 错误表现: "无法加载文件"或"不支持的MATLAB文件版本"
   - 解决方案: 
     - 检查是否安装了h5py库(`pip install h5py`)
     - 尝试使用`--verbose`参数查看详细错误信息
     - 检查processing_errors.log文件中的详细错误记录

2. **通道对齐问题**：
   - 错误表现: "无法找到足够的通道"或"插值失败"
   - 解决方案:
     - 查看原始文件的通道名称是否符合标准10-20系统
     - 考虑在CHANNEL_ALIASES字典中添加特定于您数据集的通道映射

3. **内存不足**：
   - 错误表现: "内存错误"或程序崩溃
   - 解决方案:
     - 尝试逐个处理受试者数据而非一次处理全部
     - 增加系统虚拟内存
     - 考虑在大型服务器上运行脚本

4. **数据段太短**：
   - 错误表现: "文件太短，无法分段"警告
   - 解决方案:
     - 检查原始数据的长度是否足够(至少128个采样点)
     - 考虑修改SAMPLE_LEN参数适应更短的数据片段

5. **通用错误处理**：
   - 查看`processing_errors.log`文件了解详细错误信息
   - 使用`--verbose`参数获取更详细的运行时信息
   - 检查数据格式是否与脚本期望的格式匹配

## 更多资源

- [LEAD模型官方文档](https://github.com/xxx/LEAD)
- [MNE-Python文档](https://mne.tools/stable/index.html)
- [SciPy MATLAB文件读取](https://docs.scipy.org/doc/scipy/reference/io.html)
- [h5py文档 (HDF5支持)](https://docs.h5py.org/)
- [EEGLAB文档](https://eeglab.org/tutorials/) 