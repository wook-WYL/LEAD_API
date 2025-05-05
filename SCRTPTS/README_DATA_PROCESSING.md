# EEG 数据预处理工具使用说明

本文档说明如何使用 `preprocess_public_data.py` 脚本将公开的 EEG 数据处理为 LEAD 模型可用的格式。

## 背景

LEAD 是一个用于基于 EEG 的阿尔茨海默病检测的大型基础模型。该模型需要特定格式的数据输入：

1. 数据需要重采样至 128Hz
2. 每个样本包含 128 个时间点
3. 使用标准的 10-20 系统的 19 个通道
4. 数据按受试者组织，包含 Feature 和 Label 两个文件夹
5. 数据通过频率过滤和标准化进行预处理

## 安装依赖

在使用脚本前，请先安装必要的依赖：

```bash
pip install numpy scipy mne scikit-learn
```

## 数据组织

预处理脚本假设您的原始数据按以下结构组织：

```
input_dir/
├── AD/                  # 阿尔茨海默病患者组
│   ├── subject1/        # 受试者1
│   │   ├── file1.edf    # EEG记录文件
│   │   └── file2.edf
│   └── subject2/
│       └── ...
└── Normal/              # 健康对照组
    ├── subject3/
    │   └── ...
    └── subject4/
        └── ...
```

脚本会自动检测目录名中是否包含 "ad"、"alzheimer" 或 "patient" 字符串来判断该组是否为阿尔茨海默病患者组。

## 使用方法

基本用法：

```bash
python preprocess_public_data.py --input_dir /path/to/raw/data --output_dir /path/to/output --dataset_name CUSTOM_DATASET
```

### 参数说明

- `--input_dir`：原始数据目录路径（必需）
- `--output_dir`：处理后数据的输出目录（必需）
- `--dataset_name`：数据集名称，将作为输出目录的子文件夹名（必需）
- `--original_fs`：原始数据的采样率，如果未指定则自动从数据中检测
- `--ad_class`：AD患者标签的值，默认为1
- `--normal_class`：正常人群标签的值，默认为0

### 示例

处理采样率为 500Hz 的 EEG 数据集：

```bash
python preprocess_public_data.py --input_dir /raw/my_eeg_data --output_dir /processed/dataset --dataset_name MyEEG --original_fs 500
```

## 输出格式

处理后的数据将保存在以下结构中：

```
output_dir/dataset_name/
├── Feature/
│   ├── feature_01.npy   # 受试者1的特征数据，形状为(N, 128, 19)
│   ├── feature_02.npy   # 受试者2的特征数据 
│   └── ...              # 更多受试者
└── Label/
    └── label.npy        # 所有受试者的标签，形状为(M, 2)，每行包含[标签, ID]
```

其中：
- `N` 是该受试者的样本数量
- `128` 是每个样本的时间点数量
- `19` 是通道数量
- `M` 是受试者总数
- 标签中，1通常表示AD患者，0表示健康对照组

## 后续步骤

预处理完成后，您可以将数据放入 LEAD 模型的 `dataset/` 目录下对应的数据集文件夹中，然后按照 LEAD 模型的说明进行训练或推理。

按照 LEAD 模型的数据加载器要求，您需要创建一个自定义的数据加载器，将其放在 `data_provider/dataset_loader/` 目录下。

具体步骤如下：

1. 创建文件 `data_provider/dataset_loader/custom_loader.py`
2. 参考其他数据集的加载器实现自定义加载器
3. 在 `data_provider/data_loader.py` 中引入并注册您的加载器
4. 使用 `run.py` 脚本指定您的数据集进行训练或测试

## 注意事项

1. 确保您的 EEG 数据使用标准的 10-20 系统电极位置
2. 对于电极名称有差异的数据集（如使用 T7 而非 T3），脚本会自动进行别名映射
3. 如果您的数据缺少某些必需的通道，脚本会尝试使用插值方法补充
4. 处理大型数据集可能需要较长时间和较大内存
5. 频率过滤和标准化将在 LEAD 模型的数据加载过程中进行，预处理脚本主要处理通道对齐、重采样和分段 