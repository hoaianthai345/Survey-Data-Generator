import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# ====== Likert + Regression ======
def generate_likert_data_with_regression(var_names, items_per_var, cor_matrix, y_var, betas, n_samples=300, likert_scale=5):
    var_names = np.array(var_names)
    cor_matrix = np.array(cor_matrix)
    n_vars = len(var_names)

    latent = np.random.multivariate_normal(mean=[0]*n_vars, cov=cor_matrix, size=n_samples)
    
    y_idx = var_names.tolist().index(y_var)
    x_indices = [i for i in range(n_vars) if var_names[i] != y_var]
    beta_array = np.array([betas[var_names[i]] for i in x_indices])
    noise = np.random.normal(0, 0.5, n_samples)
    latent[:, y_idx] = latent[:, x_indices] @ beta_array + noise

    all_items = {}
    for i, name in enumerate(var_names):
        for j in range(items_per_var[i]):
            raw_score = latent[:, i] + np.random.normal(0, 0.3, n_samples)
            likert = np.clip(np.round(norm.cdf(raw_score) * likert_scale), 1, likert_scale)
            col_name = f"{name}_Q{j+1}"
            all_items[col_name] = likert.astype(int)
    
    return pd.DataFrame(all_items)

# ====== Continuous Variables ======
def generate_continuous_vars(continuous_config, n_samples):
    data = {}
    for var in continuous_config:
        mu = var["mean"]
        std = var["std"]
        round_mode = var["round"]
        vals = np.random.normal(mu, std, n_samples)
        if round_mode == "int":
            vals = np.round(vals).astype(int)
        elif round_mode == "float":
            vals = np.round(vals, 2)
        elif isinstance(round_mode, int):
            vals = np.round(vals, round_mode)
        data[var["name"]] = vals
    return pd.DataFrame(data)

# ====== Streamlit UI ======
st.title("📊 Survey Data Generator")

n_vars = st.slider("🔢 Số biến tiềm ẩn (Likert)", 0, 10, 3)

var_names = []
items_per_var = []
st.subheader("📝 Cấu hình biến tiềm ẩn")

for i in range(n_vars):
    col1, col2 = st.columns([2, 1])
    name = col1.text_input(f"Tên biến {i+1}", value=f"Var{i+1}", key=f"name_{i}")
    items = col2.number_input(f"Số item", min_value=1, max_value=10, value=3, key=f"item_{i}")
    var_names.append(name)
    items_per_var.append(items)

st.subheader("🔗 Ma trận tương quan")

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

# ====== Continuous Config ======
st.subheader("📏 Biến định lượng")
n_cont_vars = st.slider("Số biến định lượng", 0, 10, 2)

continuous_config = []
for i in range(n_cont_vars):
    st.markdown(f"**Biến {i+1}**")
    name = st.text_input(f"Tên biến định lượng {i+1}", key=f"cont_name_{i}", value=f"ContVar{i+1}")
    mean = st.number_input(f"Trung bình (μ)", value=30.0, key=f"mean_{i}")
    std = st.number_input(f"Độ lệch chuẩn (σ)", value=5.0, key=f"std_{i}")
    round_mode = st.selectbox(f"Làm tròn giá trị", ["int", "float", "1", "2", "3"], index=1, key=f"round_{i}")
    round_mode = int(round_mode) if round_mode.isdigit() else round_mode
    continuous_config.append({"name": name, "mean": mean, "std": std, "round": round_mode})

# ====== Generate Data ======
st.subheader("⚙️ Tuỳ chọn")
n_samples = st.slider("Số mẫu", 100, 2000, 500)
likert_scale = st.selectbox("Thang đo Likert", [5, 7], index=0)

if st.button("🚀 Sinh dữ liệu"):
    try:
        df_likert = generate_likert_data_with_regression(
            var_names, items_per_var, cor_matrix, y_var, betas, n_samples, likert_scale
        )
        df_cont = generate_continuous_vars(continuous_config, n_samples)
        df = pd.concat([df_likert, df_cont], axis=1)

        st.success("✅ Dữ liệu đã được tạo thành công!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải xuống CSV", data=csv, file_name="survey_data.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")

st.markdown("🥰 Made by AHT")
st.markdown("BIDV - 6150845123")