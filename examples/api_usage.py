import requests
import json

BASE_URL = "http://localhost:8000"

# 1. 获取可用数据集列表
def get_available_datasets():
    response = requests.get(f"{BASE_URL}/available_datasets")
    return response.json()

# 2. 训练模型
def train_model():
    training_config = {
        "method": "LEAD",
        "task_name": "supervised",
        "model": "LEAD",
        "model_id": "test1",
        "training_datasets": "ADFTD,CNBPM",
        "testing_datasets": "ADFTD,CNBPM",
        "train_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "use_gpu": True,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2
    }
    
    response = requests.post(f"{BASE_URL}/train", json=training_config)
    return response.json()

# 3. 测试模型
def test_model():
    test_config = {
        "method": "LEAD",
        "task_name": "supervised",
        "model": "LEAD",
        "model_id": "test1",
        "training_datasets": "ADFTD,CNBPM",
        "testing_datasets": "ADFTD,CNBPM",
        "use_gpu": True,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2
    }
    
    response = requests.post(f"{BASE_URL}/test", json=test_config)
    return response.json()

if __name__ == "__main__":
    # 获取可用数据集
    print("获取可用数据集：")
    datasets = get_available_datasets()
    print(json.dumps(datasets, indent=2, ensure_ascii=False))
    
    # 训练模型
    print("\n开始训练模型：")
    train_results = train_model()
    print(json.dumps(train_results, indent=2, ensure_ascii=False))
    
    # 测试模型
    print("\n开始测试模型：")
    test_results = test_model()
    print(json.dumps(test_results, indent=2, ensure_ascii=False))