import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    加载保存的模型
    :param model_path: 模型文件路径
    :return: 加载的模型对象
    """
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")


def evaluate_model(model, X_test, y_test):
    """
    使用测试集评估模型
    :param model: 已加载的模型对象
    :param X_test: 测试集特征
    :param y_test: 测试集目标变量
    :return: 测试集评估指标和预测结果
    """
    # 使用测试集进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"MSE": mse, "MAE": mae, "R²": r2}, y_pred


def plot_predictions(y_test, y_pred):
    """
    绘制真实值与预测值的对比图
    :param y_test: 测试集真实目标值
    :param y_pred: 测试集预测值
    :return: Matplotlib figure 对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建新的图形对象
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linewidth=2,
    )
    ax.set_title("True vs Predicted Values")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    return fig  # 返回 figure 对象