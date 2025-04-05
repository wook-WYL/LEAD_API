import requests
import json

BASE_URL = "http://localhost:8000"

# 测试模型
def test_model():
    test_config = {
        "checkpoints_path": "./checkpoints/LEAD/pretrain_lead/LEAD/P-11-Base/",
        "testing_datasets": "ADFTD",
        "root_path": './dataset/'
    }
    
    response = requests.post(f"{BASE_URL}/test", json=test_config)
    return response.json()

if __name__ == "__main__":
    # 测试模型
    print("开始测试模型：")
    test_results = test_model()
    print(json.dumps(test_results, indent=2, ensure_ascii=False))