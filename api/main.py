# 导入必要的库
from operator import is_
from fastapi import FastAPI, HTTPException  # FastAPI框架和HTTP异常处理
from pydantic import BaseModel  # 用于数据验证和设置
from typing import List, Optional  # 类型提示
import sys
import os
# 添加项目根目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import train_model, test_model  # 从run模块导入训练和测试函数
from fastapi import Query

# 创建FastAPI应用实例，设置标题
app = FastAPI(title="EEG Model API")

# 定义训练配置的数据模型，使用Pydantic进行数据验证
class TrainingConfig(BaseModel):
    method: str = 'LEAD'  # 训练方法
    task_name: str = 'pretrain_lead'  # 任务名称
    model: str = 'LEAD'  # 模型类型
    model_id: str = 'P-11-Base'  # 模型ID
    is_training: int = 0  # 训练标志0表示测试，1表示训练
    data: str = 'MultiDatasets'  # 数据类型
    root_path: str = './dataset/'  # 数据集根路径
    pretraining_datasets: str = 'ADSZ,APAVA-19,ADFSU,AD-Auditory,REEG-PD-19,PEARL-Neuro-19,Depression-19,REEG-SRM-19'  # 预训练数据集
    training_datasets: str = 'ADFTD'  # 训练数据集
    testing_datasets: str = 'ADFTD'  # 测试数据集
    checkpoints_path: str = './checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/'  # 检查点路径
    e_layers: int = 12  # 编码器层数
    batch_size: int = 512  # 批量大小
    n_heads: int = 8  # 注意力头数
    d_model: int = 128  # 模型维度
    d_ff: int = 256  # 全连接层维度
    swa: bool = True  # 是否使用SWA
    des: str = 'Exp'  # 实验描述
    itr: int = 5  # 实验次数
    learning_rate: float = 0.0002  # 学习率
    train_epochs: int = 60  # 训练轮次
    top_k: int = 5
    num_kernels: int = 6
    seq_len: int = 96
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    moving_avg: int = 25
    factor: int = 1
    distil: bool = True
    dropout: float = 0.1
    embed: str = 'timeF'
    activation: str = 'gelu'
    output_attention: bool = False
    patch_len: int = 32
    stride: int = 8
    patch_len_list: str = '4'
    up_dim_list: str = '76'
    augmentations: str = 'flip,frequency,jitter,mask,channel,drop'
    no_inter_attn: bool = False
    no_temporal_block: bool = False
    no_channel_block: bool = False
    K: int = 65536
    momentum: float = 0.999
    temperature: float = 0.07
    mask_ratio: float = 0.5
    contrastive_loss: str = 'all'
    num_workers: int = 0
    patience: int = 3
    loss: str = 'MSE'
    lradj: str = 'type1'
    use_amp: bool = False
    no_normalize: bool = False
    sampling_rate: int = 128
    low_cut: float = 0.5
    high_cut: float = 45
    cross_val: str = 'fixed'
    use_gpu: bool = True
    gpu: int = 0
    use_multi_gpu: bool = True
    devices: str = '0'
    p_hidden_dims: List[int] = [128, 128]
    p_hidden_layers: int = 2

# 定义训练接口，POST方法
@app.post("/train")
async def train(config: TrainingConfig):
    try:
        # 将配置转换为字典并添加训练标志
        args_dict = config.dict()
        args_dict["is_training"] = 1
        
        # 调用训练函数
        results = train_model(args_dict)
        
        # 返回成功响应和训练结果
        return {
            "status": "success",
            "message": "模型训练完成",
            "results": results
        }
    except Exception as e:
        # 打印详细错误信息
        print(f"训练错误: {str(e)}")
        # 捕获异常并返回500错误
        raise HTTPException(status_code=500, detail=str(e))

# 定义测试接口，POST方法
@app.post("/test")
async def test(config: TrainingConfig):
    try:
        # 将配置转换为字典并添加测试标志
        args_dict = config.dict()
        args_dict["is_training"] = 0
        
        # 调用测试函数
        results = test_model(args_dict)
        print(results)
        # 返回成功响应和测试结果
        return {
            "status": "success",
            "message": "模型测试完成",
            "results": results
        }
    except Exception as e:
        # 打印详细错误信息
        print(f"测试错误: {str(e)}")
        # 捕获异常并返回500错误
        raise HTTPException(status_code=500, detail=str(e))

# 定义获取可用数据集接口，GET方法
@app.get("/available_datasets")
async def get_datasets(pretraining_path: str = Query(...), training_path: str = Query(...)):
    # 动态获取预训练数据集列表
    pretraining_datasets = [name for name in os.listdir(pretraining_path) if os.path.isdir(os.path.join(pretraining_path, name))]
    
    # 动态获取训练数据集列表
    training_datasets = [name for name in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, name))]
    
    # 返回数据集列表
    return {
        "pretraining_datasets": pretraining_datasets,
        "training_datasets": training_datasets
    }