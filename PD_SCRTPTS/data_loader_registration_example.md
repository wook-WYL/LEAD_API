# 如何在data_loader.py中注册自定义数据加载器

以下是将自定义数据加载器注册到LEAD模型系统的步骤说明。

## 1. 编辑data_provider/data_loader.py文件

首先，需要在data_loader.py文件的开头导入您的自定义加载器：

```python
# 导入现有加载器（保留原有导入）
from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
# ...其他加载器导入...

# 导入自定义加载器
from data_provider.dataset_loader.custom_loader import CUSTOMLoader
```

## 2. 在数据集字典中注册您的加载器

在`data_folder_dict`字典中添加您的数据集名称和对应的加载器类：

```python
data_folder_dict = {
    # 现有数据集（保留原有映射）
    'APAVA': APAVALoader,
    'Cognision-ERP': COGERPLoader,
    # ...其他数据集...
    
    # 添加您的自定义数据集
    'CUSTOM': CUSTOMLoader,  # 自定义数据集加载器
    'CUSTOM-19': CUSTOMLoader,  # 如果需要使用19通道配置，也可以添加该映射
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

# 导入自定义加载器
from data_provider.dataset_loader.custom_loader import CUSTOMLoader

# 数据集名称到数据加载器的映射字典
data_folder_dict = {
    # 原始通道数的数据集（用于单数据集监督学习）
    'APAVA': APAVALoader,
    'Cognision-ERP': COGERPLoader,
    # ...其他数据集保持不变...
    
    # 添加自定义数据集
    'CUSTOM': CUSTOMLoader,  # 自定义数据集加载器
}
```

## 4. 使用自定义数据集

注册完加载器后，您可以在使用`run.py`脚本时指定您的数据集名称：

```bash
# 在您的数据集上进行监督学习
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-CUSTOM-Sup --model LEAD --data SingleDataset \
--training_datasets CUSTOM \
--testing_datasets CUSTOM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# 或者在您的数据集上微调预训练模型
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/ --task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-F-CUSTOM-Base --model LEAD --data MultiDatasets \
--training_datasets CUSTOM \
--testing_datasets CUSTOM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15
```

## 注意事项

1. 确保您的数据集名称与数据集目录名称一致，这样系统才能正确找到数据
2. 目录结构应为：`dataset/CUSTOM/Feature/` 和 `dataset/CUSTOM/Label/`
3. 如果您的数据集经过了19通道对齐处理，建议也注册一个带`-19`后缀的版本（如`CUSTOM-19`）
4. 在使用多数据集训练时，应该使用`MultiDatasets`加载器，单数据集训练时使用`SingleDataset`加载器 