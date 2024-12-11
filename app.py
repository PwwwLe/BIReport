import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.data_preprocessing import load_data, basic_info, data_preprocessing
from src.data_split import split_data, display_and_download
from src.model_testing import evaluate_model, plot_predictions
from src.model_training import train_model

# 配置 streamlit 页面
st.set_page_config(
    page_title="2022302111327-彭文乐-课程报告-基于机器学习的电商用户购买行为预测：以Black Friday Sales数据集为例",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置页面标题
st.title("数据集: Black Friday Sales 展示与预处理")


# 加载数据
@st.cache_data  # 缓存数据加载结果
def load_dataset(file_path):
    return load_data(file_path)


@st.cache_data
def load_default_dataset():
    return load_data("./Data/train.csv")

st.sidebar.header("目录")
st.sidebar.markdown("""
- [1. 数据集预览](#数据集预览)
- [2. 数据集基本信息](#数据集基本信息)
- [3. 数据集预处理](#数据集预处理)
- [4. 数据集划分](#数据集划分)
- [5. 模型训练](#模型训练)
- [6. 模型测试](#模型测试)
""", unsafe_allow_html=True)

def display():
    # 展示数据集
    st.markdown('<a name="数据集预览"></a>', unsafe_allow_html=True)
    st.header("1. 数据集预览", divider=True)
    st.dataframe(data)

    st.divider()

    # 展示数据集的基本信息
    st.markdown('<a name="数据集基本信息"></a>', unsafe_allow_html=True)
    st.header("2. 数据集基本信息", divider=True)
    basic_info(data)

    st.divider()

    # 数据预处理
    st.markdown('<a name="数据集预处理"></a>', unsafe_allow_html=True)
    st.header("3. 数据集预处理", divider=True)
    data_preprocessing(data)

    st.divider()

    # 数据集划分
    st.markdown('<a name="数据集划分"></a>', unsafe_allow_html=True)
    st.header("4. 数据集划分", divider=True)
    st.subheader("划分比例调整")
    train_size = st.slider("训练集比例", 0.5, 0.8, 0.7, 0.05)
    val_size = st.slider("验证集比例", 0.1, 0.3, 0.15, 0.05)
    test_size = 1 - train_size - val_size
    st.write(f"训练集: {train_size * 100:.1f}%, 验证集: {val_size * 100:.1f}%, 测试集: {test_size * 100:.1f}%")
    # 确保比例合法
    if train_size + val_size + test_size != 1:
        st.error("训练集、验证集和测试集的比例和应该为1.")
    else:
        # 数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, "Purchase", train_size, val_size,
                                                                    test_size)

        # 展示划分结果
        st.subheader("划分后的数据集大小: ")
        st.write(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        st.subheader("划分后的数据集: ")
        display_and_download("训练集特征 (X_train)", X_train, "X_train.csv")
        display_and_download("训练集目标变量 (y_train)", pd.DataFrame(y_train, columns=["Purchase"]), "y_train.csv")
        display_and_download("验证集特征 (X_val)", X_val, "X_val.csv")
        display_and_download("验证集目标变量 (y_val)", pd.DataFrame(y_val, columns=["Purchase"]), "y_val.csv")
        display_and_download("测试集特征 (X_test)", X_test, "X_test.csv")
        display_and_download("测试集目标变量 (y_test)", pd.DataFrame(y_test, columns=["Purchase"]), "y_test.csv")

    st.divider()

    # 模型训练
    st.markdown('<a name="模型训练"></a>', unsafe_allow_html=True)
    st.header("5. 模型训练", divider=True)
    st.subheader("模型选择")
    model_type = st.selectbox("选择模型类型", ["Random Forest", "Linear Regression", "XGBoost"])
    # 超参数调整
    st.subheader("超参数调整")
    params = {}
    if model_type == "Random Forest":
        params["n_estimators"] = st.number_input("随机森林树的数量 (n_estimators)", min_value=10, max_value=500,
                                                 value=100)
        params["max_depth"] = st.number_input("最大深度 (max_depth)", min_value=1, max_value=50, value=10)
    elif model_type == "XGBoost":
        params["n_estimators"] = st.number_input("XGBoost 树的数量 (n_estimators)", min_value=10, max_value=500,
                                                 value=100)
        params["learning_rate"] = st.slider("学习率 (learning_rate)", min_value=0.01, max_value=0.3, value=0.1,
                                            step=0.01)
        params["max_depth"] = st.number_input("最大深度 (max_depth)", min_value=1, max_value=50, value=6)
    if st.button("训练模型"):
        with st.spinner("正在训练模型..."):
            model, metrics, y_val_pred = train_model(X_train, y_train, X_val, y_val, model_type=model_type, params=params)
            # 将训练好的模型存储到 session_state
            st.session_state["trained_model"] = model
            st.session_state["metrics"] = metrics
            st.session_state["y_val_pred"] = y_val_pred

        st.success("模型训练完成！")

        # 显示评估指标
        st.write("### 模型评估指标")
        st.write(f"MSE: {metrics['MSE']}")
        st.write(f"MAE: {metrics['MAE']}")
        st.write(f"R²: {metrics['R²']}")

        # 可视化评估图表
        st.write("### 模型评估图表")
        fig, ax = plt.subplots()
        ax.scatter(y_val, y_val_pred, edgecolor='k', alpha=0.7)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2)
        ax.set_title("True vs Predicted Values")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)

        # 提供模型下载
        with open("model.pkl", "rb") as file:
            st.download_button(
                label="下载训练好的模型",
                data=file,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

    st.divider()

    # 模型测试
    st.markdown('<a name="模型测试"></a>', unsafe_allow_html=True)
    st.header("6. 模型测试",divider=True)
    if "trained_model" in st.session_state:
        if st.button("测试模型"):
            with st.spinner("正在测试模型..."):
                metrics, y_pred = evaluate_model(
                    st.session_state["trained_model"], X_test, y_test
                )
                st.session_state["test_metrics"] = metrics
                st.session_state["test_predictions"] = y_pred

            st.success("模型测试完成！")

            # 显示评估指标
            st.write("### 测试集评估指标")
            st.write(f"MSE: {metrics['MSE']}")
            st.write(f"MAE: {metrics['MAE']}")
            st.write(f"R²: {metrics['R²']}")

            # 显示预测图表
            st.write("### 真实值与预测值对比图")
            fig = plot_predictions(y_test, y_pred)
            st.pyplot(fig)

data = load_default_dataset()
st.sidebar.info("默认数据集: `./data/train.csv` 已加载")
# 数据加载
st.sidebar.header("数据集加载选项")
st.sidebar.info("默认数据集: `./data/train.csv`")
uploaded_file = st.sidebar.file_uploader("上传新的数据集文件", type="csv")
if uploaded_file:
    # 读取上传的文件
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("新数据集已成功加载！")
    display()
else:
    st.sidebar.info("请上传一个 CSV 数据集文件以查看其基本信息。")

display()
