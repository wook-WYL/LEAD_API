import argparse
import os
import torch
from exp.exp_supervised import Exp_Supervised
from exp.exp_finetune import Exp_Finetune
from exp.exp_pretrain.exp_pretrain_lead import Exp_Pretrain_LEAD
from exp.exp_pretrain.exp_pretrain_moco import Exp_Pretrain_MOCO
from exp.exp_pretrain.exp_pretrain_ts2vec import Exp_Pretrain_TS2Vec
from exp.exp_pretrain.exp_pretrain_biot import Exp_Pretrain_BIOT
from exp.exp_pretrain.exp_pretrain_eeg2rep import Exp_Pretrain_EEG2Rep
import random
import numpy as np
from utils.tools import compute_avg_std

def train_model(args_dict):
    """
    训练模型的函数接口
    args_dict: 包含所有训练参数的字典
    """
    args = argparse.Namespace(**args_dict)
    # 确保包含itr属性
    args.itr = args_dict.get('itr', 1)
    args.use_multi_gpu = args_dict.get('use_multi_gpu', False)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 选择实验类型
    if args.task_name == 'supervised':
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain_lead':
        Exp = Exp_Pretrain_LEAD
    elif args.task_name == 'pretrain_moco':
        Exp = Exp_Pretrain_MOCO
    elif args.task_name == 'pretrain_ts2vec':
        Exp = Exp_Pretrain_TS2Vec
    elif args.task_name == 'pretrain_biot':
        Exp = Exp_Pretrain_BIOT
    elif args.task_name == 'pretrain_eeg2rep':
        Exp = Exp_Pretrain_EEG2Rep
    elif args.task_name == 'finetune':
        Exp = Exp_Finetune
    else:
        raise ValueError('task_name unknown')

    metrics_lists = {
        'sample_val': [], 'subject_val': [],
        'sample_test': [], 'subject_test': []
    }
    
    for ii in range(args.itr):
        seed = 41 + ii
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if args.model != "TCN":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
        args.seed = seed
        setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
            args.n_heads,
            args.e_layers,
            args.d_model,
            args.d_ff,
            args.seed
        )
    
        exp = Exp(args)
        exp.train(setting)
        metrics = exp.test(setting)
        
        metrics_lists['sample_val'].append(metrics[0])
        metrics_lists['subject_val'].append(metrics[1])
        metrics_lists['sample_test'].append(metrics[2])
        metrics_lists['subject_test'].append(metrics[3])
        torch.cuda.empty_cache()
    
    final_metrics = compute_avg_std(args, 
        metrics_lists['sample_val'],
        metrics_lists['subject_val'],
        metrics_lists['sample_test'],
        metrics_lists['subject_test'],
        metrics[4]
    )
    
    return final_metrics

def test_model(args_dict):
    """
    测试模型的函数接口
    args_dict: 包含所有测试参数的字典
    返回: 包含测试指标平均值的字典
    """
    # 将参数字典转换为Namespace对象
    args = argparse.Namespace(**args_dict)
    # 确保包含itr属性，默认为1
    args.itr = args_dict.get('itr', 1)
    args.use_multi_gpu = args_dict.get('use_multi_gpu', False)
    # 检查GPU可用性并设置use_gpu标志
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 多GPU设置处理
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]  # 主GPU设为第一个设备

    # 根据任务类型选择对应的实验类
    if args.task_name == 'supervised':
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain_lead':
        Exp = Exp_Pretrain_LEAD
    elif args.task_name == 'pretrain_moco':
        Exp = Exp_Pretrain_MOCO
    elif args.task_name == 'pretrain_ts2vec':
        Exp = Exp_Pretrain_TS2Vec
    elif args.task_name == 'pretrain_biot':
        Exp = Exp_Pretrain_BIOT
    elif args.task_name == 'pretrain_eeg2rep':
        Exp = Exp_Pretrain_EEG2Rep
    elif args.task_name == 'finetune':
        Exp = Exp_Finetune
    else:
        raise ValueError('task_name unknown')

    # 初始化指标存储字典
    metrics_lists = {
        'sample_val': [], 'subject_val': [],  # 验证集指标
        'sample_test': [], 'subject_test': []  # 测试集指标
    }
    
    # 进行多次实验迭代(itr次)
    for ii in range(args.itr):
        # 设置随机种子保证可重复性
        seed = 41 + ii
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 非TCN模型设置确定性计算
        if args.model != "TCN":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
        args.seed = seed
        # 生成实验设置标识字符串
        setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
            args.n_heads,
            args.e_layers,
            args.d_model,
            args.d_ff,
            args.seed
        )
    
        # 初始化实验对象
        exp = Exp(args)
        print(f"Testing with setting: {setting}")  # 日志输出当前设置
        # 执行测试并获取指标
        metrics = exp.test(setting, test=1)
        #print("测试exp环节")
       #print(metrics)
        
        # 存储当前迭代的指标
        
        metrics_lists['sample_val'].append(metrics[0])
        metrics_lists['subject_val'].append(metrics[1])
        metrics_lists['sample_test'].append(metrics[2])
        metrics_lists['subject_test'].append(metrics[3])
        torch.cuda.empty_cache()  # 清理GPU缓存
    
    print("测试完成环节")
    #print(metrics_lists)
    # 计算所有迭代的平均指标
    final_metrics = compute_avg_std(args, 
        metrics_lists['sample_val'],
        metrics_lists['subject_val'],
        metrics_lists['sample_test'],
        metrics_lists['subject_test'],
        metrics[4]  # 其他指标
    )
    
    #print(f"Final metrics: {final_metrics}")  # 输出最终指标
    print("完成")  # 测试完成日志
    #print(final_metrics)
    return final_metrics  # 返回最终指标结果

if __name__ == '__main__':

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='LEAD')

    # 基本配置参数
    parser.add_argument('--method', type=str, required=True, default='LEAD',
                        help='Overall method (combinations of task_name, model, model_id) name, '
                             'options: [LEAD, MOCO, Transformer, TCN]')
    parser.add_argument('--task_name', type=str, required=True, default='supervised',
                        help='task name, options:[supervised, pretrain_lead, pretrain_moco, finetune]')
    parser.add_argument('--model', type=str, required=True, default='LEAD',
                        help='backbone model name, options: [Transformer, TCN, LEAD]')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    # 数据加载参数
    parser.add_argument('--data', type=str, required=True, default='Single-Dataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of all dataset folders')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument("--pretraining_datasets", type=str, default="ADSZ",
                        help="List of datasets folder names for pretraining (No overlapping with downstream datasets).")
    parser.add_argument("--training_datasets", type=str, default="ADFTD",
                        help="List of datasets folder names for pretraining linear probe, supervised, and finetune training.")
    parser.add_argument("--testing_datasets", type=str, default="ADFTD",
                        help="List of datasets folder names for pretraining linear probe, supervised, and finetune validation and test.")
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/',
                        help='location of pre-trained model checkpoints')

    # 模型定义参数
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=32, help='patch_len used in PatchTST, BIOT')
    parser.add_argument('--stride', type=int, default=8, help='stride used in PatchTST')

    # ADformer/LEAD 特定参数
    parser.add_argument("--patch_len_list", type=str, default="4", help="a list of patch len used in Medformer, ADformer")
    parser.add_argument("--up_dim_list", type=str, default="76",
                        help="a list of up dimension factor used in ADformer")
    parser.add_argument("--augmentations", type=str, default="flip,frequency,jitter,mask,channel,drop",
                        help="a comma-seperated list of augmentation types (none, jitter or scale). "
                             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",)
    parser.add_argument("--no_inter_attn", action="store_true",
                        help="whether to use inter-attention in encoder, "
                             "using this argument means not using inter-attention", default=False)
    parser.add_argument("--no_temporal_block", action="store_true",
                        help="whether to use temporal block in encoder", default=False)
    parser.add_argument("--no_channel_block", action="store_true",
                        help="whether to use channel block in encoder", default=False)

    # MOCO 特定参数
    parser.add_argument('--K', type=int, default=65536, help='Size of the queue in MOCO method')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum for updating the key encoder')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for the softmax in MOCO method')

    # EEG2Rpe 特定参数
    parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")

    # LEAD 特定参数
    parser.add_argument('--contrastive_loss', type=str, default='all',
                        help='contrastive loss modules enabled, options: [sample,subject,all]')

    # 优化参数
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument("--swa", action="store_true", help="use stochastic weight averaging", default=False)
    parser.add_argument('--no_normalize', action='store_true',
                        help='do not normalize data in data loader', default=False)
    parser.add_argument('--sampling_rate', type=int, default=128, help='frequency sampling rate')
    parser.add_argument('--low_cut', type=float, default=0.5, help='low cut for bandpass filter')
    parser.add_argument('--high_cut', type=float, default=45, help='high cut for bandpass filter')
    parser.add_argument('--cross_val', type=str, default='fixed',
                        help='cross validation methods, options: [fixed, mccv, loso]')

    # GPU 参数
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')

    # 去稳定投影器参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # 解析参数
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # 如果是直接运行这个文件，则执行训练
    if args.is_training:
        final_metrics = train_model(vars(args))
    else:
        final_metrics = test_model(vars(args))
