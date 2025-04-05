import requests

url = "http://localhost:8000/train"
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
    "e_layers": 2,
    "dataset_path": "D:\\ZILIAO\\programstudy\\EEG\\LEAD\\dataset\\feature_01.npy" 
}
print("\n开始训练模型：")
response = requests.post(url, json=training_config)
print(response.json())