import io

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    加载数据集
    :param file_path: 数据集路径
    :return: DataFrame 对象
    """
    try:
        data = pd.read_csv(file_path)
        st.sidebar.info("数据加载成功!")
        return data
    except FileNotFoundError:
        st.sidebar.error(f"文件加载错误: {file_path}")
        return None


def basic_info(data):
    """
    在 Streamlit 中展示数据集的基本信息
    :param data: DataFrame 对象
    """
    # 数据结构和列信息
    st.subheader("**数据集的形状 (行数, 列数)：**")
    st.write(data.shape)

    # 数据集的列信息
    st.subheader("**数据集的列信息：**", )
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()

    # 提取列信息为 DataFrame
    columns_info = {
        "列名": data.columns,
        "非空值计数": [data[col].notnull().sum() for col in data.columns],
        "数据类型": [data[col].dtype for col in data.columns]
    }
    columns_df = pd.DataFrame(columns_info)
    st.table(columns_df)

    # 数值型特征的统计摘要
    st.subheader("**数值型特征的统计摘要：**")
    st.table(data.describe())

    # 缺失值统计
    st.subheader("**缺失值统计：**")
    st.table(data.isnull().sum())

    # 重复值统计
    st.subheader("**重复值统计：**", )
    st.write(f"重复行的数量：{data.duplicated().sum()}")

    # 目标变量基本信息
    if 'Purchase' in data.columns:
        st.subheader("**目标变量 'Purchase' 的统计摘要：**")
        st.table(data['Purchase'].describe())


def check_unique_values(data):
    """
    检查分类变量的唯一值
    :param data: DataFrame 对象
    """
    # 定义需要检查的分类变量列
    categorical_columns = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years',
                           'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    print("========== 分类变量的唯一值统计 ==========")
    for col in categorical_columns:
        if col in data.columns:
            unique_values = data[col].unique()  # 获取唯一值
            print(f"\n{col} 的唯一值：")
            print(unique_values)
            print(f"{col} 的唯一值数量：{len(unique_values)}")
            print(f"{col} 的值计数：")
            print(data[col].value_counts())
        else:
            print(f"\n警告：列 {col} 不在数据集中")


def handle_missing_values(data):
    """
    缺失值处理
    :param data: DataFrame 对象
    :return: 缺失值处理后的 DataFrame
    """
    # 打印每列缺失值数量
    missing_values = data.isnull().sum()
    st.text("缺失值统计：")
    st.table(missing_values[missing_values > 0])

    # 处理数值型特征
    # 填充 Product_Category_2 和 Product_Category_3 的缺失值为 0
    data['Product_Category_2'].fillna(0, inplace=True)
    data['Product_Category_3'].fillna(0, inplace=True)

    st.success("缺失值处理完成！")
    return data


def encode_features(data):
    """
    特征值编码
    :param data:  DataFrame 对象
    :return:  特征编码后的 DataFrame
    """
    # 性别编码
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
    # 年龄区间编码
    if 'Age' in data.columns:
        age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
        data['Age'] = data['Age'].map(age_mapping)

    # 城市类别编码
    if 'City_Category' in data.columns:
        city_mapping = {'A': 1, 'B': 2, 'C': 3}
        data['City_Category'] = data['City_Category'].map(city_mapping)

    # 当前城市居住年数处理
    if 'Stay_In_Current_City_Years' in data.columns:
        data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].str.replace('+', '').astype(int)
    st.success("特征编码完成！")
    return data


def drop_unused_columns(data):
    """
    删除无用列
    :param data: DataFrame 对象
    :return: 删除无用列后的 DataFrame
    """
    # 删除列
    existing_columns = [col for col in ['User_ID', 'Product_ID'] if col in data.columns]
    if existing_columns:
        data.drop(columns=existing_columns, inplace=True)
        st.success(f"已删除以下列：{existing_columns}")
    else:
        st.error("未找到要删除的列。")

    st.success("无用列删除完成！")
    return data


def standardize_data(data):
    """
    数据标准化
    :param data: DataFrame 对象
    :return:
    """
    # 检查列是否存在
    existing_columns = [col for col in ['Purchase'] if col in data.columns]
    if not existing_columns:
        st.warning("未找到需要标准化的列。")
        return data
    scaler = StandardScaler()
    # 对列进行标准化
    data[existing_columns] = scaler.fit_transform(data[existing_columns])
    st.success("标准化完成！")
    return data


def data_preprocessing(data):
    """
    数据集预处理方法
    :param data: DataFrame 对象
    :return:
    """
    # 处理缺失值
    handle_missing_values(data)
    # 特征值编码
    encode_features(data)
    # 删除无用列
    drop_unused_columns(data)
    # 标准化数据
    standardize_data(data)

    st.subheader("预处理后的数据集: ")
    st.dataframe(data)
