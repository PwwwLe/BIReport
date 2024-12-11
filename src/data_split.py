import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(data, target_column, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    将数据划分为训练集、验证集和测试集，支持用户自定义比例
    :param data: DataFrame 数据集
    :param target_column: 目标变量的列名
    :param train_size: 训练集比例（默认0.7）
    :param val_size: 验证集比例（默认0.15）
    :param test_size: 测试集比例（默认0.15）
    :return: 分别返回训练集、验证集和测试集的特征 (X) 和目标变量 (y)
    """
    # 确保划分比例和为 1
    assert train_size + val_size + test_size == 1, "Train, Val, and Test sizes must sum to 1."

    # 特征和目标变量分离
    X = data.drop(target_column, axis=1)  # 特征
    y = data[target_column]  # 目标变量

    # 对分类变量进行独热编码
    X = pd.get_dummies(X, drop_first=True)

    # 数值型变量标准化
    scaler = StandardScaler()
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # 第一次划分：训练集 (train_size) 和临时集 (1 - train_size)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=42)

    # 第二次划分：临时集再划分为验证集和测试集
    val_ratio = val_size / (val_size + test_size)  # 在临时集中划分验证集比例
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def display_and_download(label, data, file_name):
    """
    展示数据并提供下载功能
    :param label: 数据集标签
    :param data: 数据集 DataFrame
    :param file_name: 下载文件名
    """
    st.write(f"### {label}")
    st.dataframe(data)
    st.download_button(
        label=f"下载 {file_name}",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name=file_name,
        mime="text/csv"
    )