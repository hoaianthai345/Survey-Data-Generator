# fake_multivariable_data.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Fake Data cho Mô hình Hồi quy Đa biến")

# Tham số từ người dùng
n = st.number_input("Số lượng mẫu", min_value=10, max_value=10000, value=100)
num_features = st.slider("Số lượng biến đầu vào (X)", min_value=1, max_value=10, value=3)
noise_level = st.slider("Mức độ nhiễu (noise)", 0.0, 50.0, 10.0)

# Tạo dữ liệu
np.random.seed(42)
X = np.random.normal(0, 1, (n, num_features))
beta = np.random.uniform(-5, 5, num_features)  # hệ số ngẫu nhiên
noise = np.random.normal(0, noise_level, n)
Y = X @ beta + noise

# Tạo DataFrame
columns = [f"X{i+1}" for i in range(num_features)]
df = pd.DataFrame(X, columns=columns)
df["Y"] = Y

# Hiển thị dữ liệu
st.subheader("📄 Dữ liệu đã tạo")
st.dataframe(df.head())

# Hiển thị công thức
formula = "Y = " + " + ".join([f"{beta[i]:.2f}*X{i+1}" for i in range(num_features)]) + f" + noise"
st.code(formula, language="python")

# Tải xuống CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Tải xuống CSV",
    data=csv,
    file_name="multivariable_fake_data.csv",
    mime="text/csv"
)
