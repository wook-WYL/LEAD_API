# 如何在data_loader.py中注册PD_REST数据加载器

以下是将PD_REST数据加载器注册到LEAD模型系统的步骤说明。

## 1. 编辑data_provider/data_loader.py文件

首先，需要在data_loader.py文件的开头导入PD_REST加载器：

```python
# 导入现有加载器（保留原有导入）
from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
# ...其他加载器导入...

# 导入PD_REST加载器
from data_provider.dataset_loader.pd_rest_loader import PDRESTLoader
```

## 2. 在数据集字典中注册PD_REST加载器

在`data_folder_dict`字典中添加PD_REST数据集名称和对应的加载器类：

```python
data_folder_dict = {
    # 现有数据集（保留原有映射）
    'APAVA': APAVALoader,
    'Cognision-ERP': COGERPLoader,
    # ...其他数据集...
    
    # 添加PD_REST数据集
    'PD_REST': PDRESTLoader,  # 帕金森病静息态EEG数据集
}
```

## 3. 完整示例

下面是完整的修改示例（仅显示需要修改的部分）：

```python
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

# 导入现有加载器
from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
# ...其他导入保持不变...

# 导入PD_REST加载器
from data_provider.dataset_loader.pd_rest_loader import PDRESTLoader

# 数据集名称到数据加载器的映射字典
data_folder_dict = {
    # 原始通道数的数据集（用于单数据集监督学习）
    'APAVA': APAVALoader,
    'Cognision-ERP': COGERPLoader,
    # ...其他数据集保持不变...
    
    # 添加PD_REST数据集
    'PD_REST': PDRESTLoader,  # 帕金森病静息态EEG数据集
}
```

## 4. 使用PD_REST数据集

注册完加载器后，您可以在使用`run.py`脚本时指定PD_REST数据集名称：

```bash
# 在PD_REST数据集上进行监督学习
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-PD-Sup --model LEAD --data SingleDataset \
--training_datasets PD_REST \
--testing_datasets PD_REST \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# 或者在PD_REST数据集上微调预训练模型
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-PD-Base --model LEAD --data MultiDatasets \
--training_datasets PD_REST \
--testing_datasets PD_REST \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15
```

## 5. 多数据集训练

如果您想将PD_REST数据集与其他数据集一起训练，可以使用逗号分隔的数据集列表：

```bash
# 与AD数据集一起训练
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-PD-AD-Sup --model LEAD --data MultiDatasets \
--training_datasets PD_REST,ADFTD,BrainLat-19 \
--testing_datasets PD_REST,ADFTD,BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15
```

## 注意事项

1. 确保PD_REST数据集已经通过`preprocess_pd_data.py`脚本处理完成，并保存在`dataset/PD_REST/`目录下
2. PDRESTLoader会自动处理标签映射：
   - 健康对照组(标签值0)会映射为类别0
   - 帕金森患者(标签值2)会映射为类别1
3. 加载器支持以下主要参数：
   - `scale`: 是否标准化数据（默认为True）
   - `freq_mask`: 是否应用频率过滤（默认为False）
   - `a`: 训练集比例（默认为0.8）
   - `b`: 验证集比例（默认为0.1）
   - `random_seed`: 随机划分种子（默认为42）
4. 加载器将确保类别平衡，即使用分层抽样保证训练、验证和测试集中类别比例一致

## 加载器功能概述

新实现的PDRESTLoader提供以下主要功能：

1. **自动标签映射**: 将原始标签（0表示正常，2表示PD）映射为模型友好的格式（0和1）
2. **分层抽样**: 确保训练/验证/测试集中的类别比例一致
3. **数据预处理**: 支持标准化和频率过滤
4. **灵活数据分割**: 可自定义训练、验证和测试集的比例
5. **详细日志**: 打印加载和处理过程中的详细信息，便于调试

通过以上步骤，您可以将PD_REST数据集集成到LEAD模型训练流程中，用于帕金森病的脑电图检测研究。 