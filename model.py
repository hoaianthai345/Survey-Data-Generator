import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

def generate_likert_data_with_regression(var_names, items_per_var, cor_matrix, y_var, betas, n_samples=300, likert_scale=5):
    var_names = np.array(var_names)
    cor_matrix = np.array(cor_matrix)
    n_vars = len(var_names)

    # Tạo dữ liệu các biến tiềm ẩn
    latent = np.random.multivariate_normal(mean=[0]*n_vars, cov=cor_matrix, size=n_samples)
    
    # Tính biến Y từ mô hình hồi quy
    y_idx = var_names.tolist().index(y_var)
    x_indices = [i for i in range(n_vars) if var_names[i] != y_var]
    beta_array = np.array([betas[var_names[i]] for i in x_indices])
    noise = np.random.normal(0, 0.5, n_samples)
    latent[:, y_idx] = latent[:, x_indices] @ beta_array + noise

    # Chuyển latent thành câu hỏi Likert
    all_items = {}
    for i, name in enumerate(var_names):
        for j in range(items_per_var[i]):
            raw_score = latent[:, i] + np.random.normal(0, 0.3, n_samples)
            likert = np.clip(np.round(norm.cdf(raw_score) * likert_scale), 1, likert_scale)
            all_items[f"{name}_Q{j+1}"] = likert.astype(int)
    
    return pd.DataFrame(all_items)

# ========== STREAMLIT ==========
st.title("🧠 Tạo dữ liệu Likert từ mô hình hồi quy")

n_vars = st.slider("🔢 Số biến tiềm ẩn", 2, 10, 3)

var_names = []
items_per_var = []
st.subheader("📝 Đặt tên và số câu hỏi cho từng biến")

for i in range(n_vars):
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input(f"Tên biến {i+1}", value=f"Var{i+1}", key=f"name_{i}")
    with col2:
        items = st.number_input(f"Số item cho {name}", min_value=1, max_value=10, value=3, key=f"item_{i}")
    var_names.append(name)
    items_per_var.append(items)

st.subheader("🔗 Nhập ma trận tương quan giữa các biến")
cor_matrix = []
for i in range(n_vars):
    row = []
    cols = st.columns(n_vars)
    for j in range(n_vars):
        if j < i:
            row.append(cor_matrix[j][i])
            cols[j].markdown(f"`{cor_matrix[j][i]}`")
        elif j == i:
            row.append(1.0)
            cols[j].markdown("`1.0`")
        else:
            val = cols[j].number_input(f"r({var_names[i]}, {var_names[j]})", min_value=-1.0, max_value=1.0, value=0.4, key=f"r_{i}_{j}")
            row.append(val)
    cor_matrix.append(row)

st.subheader("📉 Mô hình hồi quy")

y_var = st.selectbox("Chọn biến phụ thuộc (Y)", var_names)
betas = {}
for var in var_names:
    if var != y_var:
        betas[var] = st.number_input(f"Hệ số beta cho {var} → {y_var}", value=0.5, format="%.2f")

st.subheader("⚙️ Tùy chọn")
n_samples = st.slider("Số mẫu khảo sát", 100, 2000, 500)
likert_scale = st.selectbox("Thang đo Likert", [5, 7], index=0)

if st.button("🚀 Sinh dữ liệu"):
    try:
        df = generate_likert_data_with_regression(var_names, items_per_var, cor_matrix, y_var, betas, n_samples, likert_scale)
        st.success("🎉 Dữ liệu đã được tạo thành công!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải xuống CSV",
            data=csv,
            file_name="likert_regression_data.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo dữ liệu: {e}")
