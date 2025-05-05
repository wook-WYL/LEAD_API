## 注意事项

1. 使用相对路径时，请确保运行脚本的当前工作目录是脚本所在的目录（PD_SCRTPTS）
2. 如需在其他位置运行脚本，请相应调整路径参数
3. 如果要使用绝对路径，可以在运行脚本时通过命令行参数指定：

```bash
python preprocess_pd_data.py --input_dir /your/absolute/path/to/PD_DATASET --output_dir /your/absolute/path/to/dataset
``` 