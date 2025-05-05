import os
import shutil
import re

# 源数据目录和目标目录
source_dir = r"../PD REST"
target_dir = r"../PD_DATASET"

# 创建PD和Normal目录
pd_dir = os.path.join(target_dir, "PD")
normal_dir = os.path.join(target_dir, "Normal")

os.makedirs(pd_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)

# 正则表达式匹配文件名模式
pattern = r"(\d+)_\d+_PD_REST\d*\.mat"

# 获取所有唯一的受试者ID
subject_ids = set()
for filename in os.listdir(source_dir):
    match = re.match(pattern, filename)
    if match:
        subject_id = match.group(1)
        subject_ids.add(subject_id)

print(f"发现 {len(subject_ids)} 个受试者ID")

# 为每个受试者创建目录
for subject_id in subject_ids:
    subject_dir = os.path.join(pd_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    print(f"创建目录: {subject_dir}")

# 将mat文件复制到相应的受试者目录
file_count = 0
for filename in os.listdir(source_dir):
    match = re.match(pattern, filename)
    if match:
        subject_id = match.group(1)
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(pd_dir, subject_id, filename)
        
        # 如果是文件（非目录），则复制
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)
            file_count += 1
            print(f"复制文件: {filename} 到 {os.path.join('PD', subject_id)}")

# 复制Excel和m文件到根目录
for filename in os.listdir(source_dir):
    if filename.endswith(".xlsx") or filename.endswith(".m"):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)
            print(f"复制文件: {filename} 到目标根目录")

print("\n数据目录结构创建完成!")
print(f"共处理了 {len(subject_ids)} 个受试者，复制了 {file_count} 个.mat文件")
print(f"目标目录: {target_dir}")