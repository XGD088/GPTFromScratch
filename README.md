# GPT从零实现项目

这个项目是一个从零实现的GPT（Generative Pre-trained Transformer）模型，包括从预训练到微调的完整流程，并提供API服务。

## 项目简介

本项目实现了一个完整的GPT模型，包括：
- 基础GPT模型架构实现
- 预训练功能
- 微调功能（分类和指令微调）
- FastAPI服务接口

模型支持多种配置，包括GPT2-small（124M）、GPT2-medium（355M）、GPT2-large（774M）和GPT2-xl（1558M）。

## 环境要求

- Python 3.9+
- PyTorch 2.0.0+
- 其他依赖库（详见requirements.txt）

## 安装步骤

1. 克隆项目到本地：
```
git clone <仓库链接>
cd GPTFromScratch
```

2. 安装依赖：
```
pip install -r requirements.txt
```

3. 下载预训练模型权重（如需要）：
```
python pre_training/gpt_download.py
```

## 项目结构

```
GPTFromScratch/
├── api/                      # API服务实现
│   ├── Model.py              # 模型加载和预测逻辑
│   ├── endpoints.py          # FastAPI端点定义
│   └── instruction-data.json # 指令微调数据
├── attention_mechanism/      # 注意力机制实现
├── fine_tuning/              # 微调相关代码
│   ├── classification/       # 分类任务微调
│   └── instruction/          # 指令微调
├── pre_training/             # 预训练相关代码
│   ├── CalculateLoss.py      # 损失计算
│   ├── LoadWeightFromOpenAI.py # 加载OpenAI权重
│   ├── TrainModel.py         # 训练模型
│   └── gpt_download.py       # 下载GPT模型
├── resources/                # 资源文件
├── test/                     # 测试代码
├── text_processer/           # 文本处理
├── utils/                    # 工具函数
├── GPTConfig.py              # GPT配置文件
├── GPTModel.py               # GPT模型实现
├── Dockerfile                # Docker构建文件
└── requirements.txt          # 依赖库列表
```

## 使用方法

### 作为Python库使用

```python
from GPTModel import GPTModel
from GPTConfig import GPT2_SMALL_CONFIG

# 初始化模型
model = GPTModel(GPT2_SMALL_CONFIG)

# 加载预训练权重（如果需要）
# ...

# 使用模型进行预测
# ...
```

### 启动API服务

```
uvicorn api.endpoints:app --host 0.0.0.0 --port 8000
```

或使用Docker：

```
docker build -t gpt-from-scratch .
docker run -p 8000:8000 gpt-from-scratch
```

### API使用示例

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {"input_data": "Every effort moves"}
response = requests.post(url, json=data)
result = response.json()
print(result["prediction"])
```

## 模型训练

### 预训练

```
python pre_training/Main.py
```

### 微调

根据需要选择分类任务或指令微调：

```
python fine_tuning/classification/train.py
```

或

```
python fine_tuning/instruction/train.py
```

## 模型配置

项目支持多种GPT配置，在`GPTConfig.py`中定义：

- GPT2-small (124M): 12层，12个注意力头，768维嵌入
- GPT2-medium (355M): 24层，16个注意力头，1024维嵌入
- GPT2-large (774M): 36层，20个注意力头，1280维嵌入
- GPT2-xl (1558M): 48层，25个注意力头，1600维嵌入

