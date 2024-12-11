import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(X_train, y_train, X_val, y_val, model_type="Random Forest", params=None, model_path="model.pkl"):
    """
    训练模型并保存
    :param X_train: 训练集特征
    :param y_train: 训练集目标变量
    :param X_val: 验证集特征
    :param y_val: 验证集目标变量
    :param model_type: 模型类型 ("Random Forest", "Linear Regression", "XGBoost")
    :param params: 模型参数字典
    :param model_path: 模型保存路径
    :return: 模型、验证集评估指标、预测值
    """
    # 初始化模型
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=42
        )
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 6),
            random_state=42
        )
    else:
        raise ValueError("Unsupported model type!")

    # 训练模型
    model.fit(X_train, y_train)

    # 验证集预测
    y_val_pred = model.predict(X_val)

    # 模型评估
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    # 保存模型
    joblib.dump(model, model_path)

    return model, {"MSE": mse, "MAE": mae, "R²": r2}, y_val_pred