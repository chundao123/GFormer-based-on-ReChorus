# Replicate GFormer model based on Rechorus

本项目旨在基于`ReChorus`框架复现论文`Graph Transformer for Recommendation`
`ReChorus`仓库：https://github.com/THUwangcy/ReChorus
`Graph Transformer for Recommendation`仓库：https://github.com/HKUDS/GFormer


## Main Structure
- data/: 数据集
- log/: 模型运行结果
- src/: 模型代码
    - main.py: Rechorus提供的主程序
    - models/: 模型定义
        - general/:
            - FuckGF.py: 复现实现

## How to use
运行以下命令：
```
python main.py --model_name FuckGF --dataset Grocery_and_Gourmet_Food --num_workers 0
```