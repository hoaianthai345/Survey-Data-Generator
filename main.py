import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, poisson

st.set_page_config(page_title="Fake Data Generator", layout="centered")
st.title("📊 Ứng dụng Giả lập Dữ liệu Nghiên cứu Thống kê")

st.sidebar.header("1. Cấu hình dữ liệu")
data_type = st.sidebar.selectbox("Loại thang đo", ["Tỷ lệ", "Khoảng", "Thứ bậc", "Danh nghĩa"])
num_rows = st.sidebar.number_input("Số lượng quan sát", min_value=10, max_value=10000, value=100)

col_name = st.sidebar.text_input("Tên biến độc lập (X1)", value="X1")

st.sidebar.header("2. Phân phối và tham số X1")
dist_type = st.sidebar.selectbox("Chọn phân phối X1", ["Normal", "Uniform", "Poisson"])

if dist_type == "Normal":
    mean = st.sidebar.number_input("Mean (Trung bình)", value=50.0)
    std = st.sidebar.number_input("Std Dev (Độ lệch chuẩn)", value=10.0)
elif dist_type == "Uniform":
    min_val = st.sidebar.number_input("Giá trị nhỏ nhất", value=0.0)
    max_val = st.sidebar.number_input("Giá trị lớn nhất", value=100.0)
elif dist_type == "Poisson":
    lam = st.sidebar.number_input("Lambda (tần suất trung bình)", value=3.0)

st.sidebar.header("3. Cấu hình hồi quy")
add_regression = st.sidebar.checkbox("Thêm biến phụ thuộc (Y) với mô hình hồi quy")

if add_regression:
    st.sidebar.markdown("Nhập mô hình hồi quy dạng: Y = a + b1*X1")
    intercept = st.sidebar.number_input("Hằng số (a)", value=10.0)
    coef_x1 = st.sidebar.number_input("Hệ số X1 (b1)", value=2.0)
    noise_std = st.sidebar.number_input("Độ lệch chuẩn của nhiễu (noise)", value=5.0)

# Tạo dữ liệu khi nhấn nút
data = None
if st.sidebar.button("🚀 Tạo dữ liệu"):
    if dist_type == "Normal":
        x1 = np.random.normal(loc=mean, scale=std, size=num_rows)
    elif dist_type == "Uniform":
        x1 = np.random.uniform(low=min_val, high=max_val, size=num_rows)
    elif dist_type == "Poisson":
        x1 = np.random.poisson(lam=lam, size=num_rows)

    # Điều chỉnh theo loại thang đo
    if data_type == "Thứ bậc":
        x1 = np.round(x1)
    elif data_type == "Danh nghĩa":
        x1 = pd.qcut(x1, q=4, labels=["A", "B", "C", "D"])

    data = pd.DataFrame({col_name: x1})

    if add_regression and data_type in ["Tỷ lệ", "Khoảng"]:
        noise = np.random.normal(0, noise_std, num_rows)
        y = intercept + coef_x1 * data[col_name].astype(float) + noise
        data["Y"] = y

    st.success("✅ Dữ liệu đã được tạo!")
    st.dataframe(data.head())

    # Tải về
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Tải xuống CSV", data=csv, file_name="fake_data.csv", mime="text/csv")
