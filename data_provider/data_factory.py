# 导入必要的模块
from data_provider.data_loader import MultiDatasetsLoader  # 多数据集加载器
from data_provider.data_loader import SingleDatasetLoader  # 单数据集加载器
from data_provider.uea import collate_fn  # 数据整理函数
from torch.utils.data import DataLoader  # PyTorch数据加载器
from utils.tools import CustomGroupSampler  # 自定义分组采样器

# 定义数据加载器映射字典
data_type_dict = {
    # 单数据集加载配置
    'SingleDataset': SingleDatasetLoader,

    # 多数据集加载配置（将多个数据集拼接）
    'MultiDatasets': MultiDatasetsLoader,  # 数据集文件夹名在args.data_folder_list中指定
}

def data_provider(args, flag):
    """数据提供函数，根据任务类型返回对应的数据加载器
    
    Args:
        args: 包含所有配置参数的对象
        flag: 数据标志（train/test/val等）
    """
    # 根据数据配置选择对应的数据加载器类
    Data = data_type_dict[args.data]

    # 测试数据配置
    if flag == 'test':
        shuffle_flag = False  # 测试时不打乱数据
        drop_last = True  # 丢弃最后不足batch_size的数据
        # 判断任务类型决定batch_size
        if args.task_name == 'supervised'\
                or args.task_name == 'pretrain_lead' \
                or args.task_name == 'pretrain_moco' \
                or args.task_name == 'pretrain_ts2vec' \
                or args.task_name == 'pretrain_biot' \
                or args.task_name == 'pretrain_eeg2rep' \
                or args.task_name == 'finetune':
            batch_size = args.batch_size  # 使用配置的batch_size
        else:
            batch_size = 1  # 评估时默认batch_size=1
    # 训练/验证数据配置
    else:
        shuffle_flag = True  # 训练时打乱数据
        drop_last = True  # 丢弃最后不足batch_size的数据
        batch_size = args.batch_size  # 使用配置的batch_size

    # 处理监督学习/非LEAD预训练/微调任务的数据加载
    if args.task_name == 'supervised' \
            or args.task_name == 'pretrain_moco' \
            or args.task_name == 'pretrain_ts2vec' \
            or args.task_name == 'pretrain_biot' \
            or args.task_name == 'pretrain_eeg2rep' \
            or args.task_name == 'finetune':
        drop_last = False  # 不丢弃最后不足batch_size的数据
        # 创建数据集实例
        data_set = Data(
            root_path=args.root_path,  # 数据集根路径
            args=args,  # 配置参数
            flag=flag,  # 数据标志
        )

        # 创建标准数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,  # 多线程加载
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # 数据整理函数（仅在生成batch时调用）
        )
        return data_set, data_loader

    # 处理LEAD预训练任务的数据加载
    elif args.task_name == 'pretrain_lead':
        # 创建数据集实例
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        # 使用自定义分组采样器（每组2个样本）
        sampler = CustomGroupSampler(data_set, batch_size=batch_size, group_size=2)
        # 创建带分组采样的数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,  # 使用自定义采样器
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # 数据整理函数
        )

        return data_set, data_loader
