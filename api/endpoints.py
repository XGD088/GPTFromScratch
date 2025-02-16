# src/my_package/api/endpoints.py
from fastapi import FastAPI
from pydantic import BaseModel

from api.Model import predict_text

# 初始化 FastAPI 应用
app = FastAPI()

class InputData(BaseModel):
    input_data: str


# 初始化模型

@app.post("/predict")
def predict(data: InputData):
    """
    预测接口
    :param input_data: 输入数据
    :return: 预测结果
    """
    result = predict_text(data.input_data)
    return {"prediction": result}