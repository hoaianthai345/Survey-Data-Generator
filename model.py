import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# ======== XỬ LÝ DỮ LIỆU ========
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

def generate_continuous_vars(config, n_samples):
    data = {}
    for var in config:
        mu, std, round_mode = var["mean"], var["std"], var["round"]
        vals = np.random.normal(mu, std, n_samples)
        if round_mode == "int":
            vals = np.round(vals).astype(int)
        elif round_mode == "float":
            vals = np.round(vals, 2)
        elif isinstance(round_mode, int):
            vals = np.round(vals, round_mode)
        data[var["name"]] = vals
    return pd.DataFrame(data)

def generate_categorical_vars(config, n_samples):
    data = {}
    for var in config:
        categories = var["categories"]
        probs = var["probs"]
        values = np.random.choice(categories, size=n_samples, p=probs)
        data[var["name"]] = values
    return pd.DataFrame(data)

# ======== GIAO DIỆN ỨNG DỤNG ========
st.set_page_config(page_title="Survey Data Generator", layout="wide")
st.title("📊 Survey Data Generator")
st.markdown("""
### 🎯 Mục đích:
Ứng dụng hỗ trợ tạo **dữ liệu khảo sát mô phỏng** dùng trong nghiên cứu định lượng.

### 🛠️ Cách sử dụng:
1. **Tạo biến tiềm ẩn (Likert)** và khai báo ma trận tương quan.
2. Khai báo **mô hình hồi quy** (chọn biến phụ thuộc và nhập hệ số).
3. Tùy chọn thêm **biến định lượng** và **biến định tính**.
4. Nhấn nút `🚀 Sinh dữ liệu` để tạo bảng và tải về dạng CSV.
""")

with st.expander("1️⃣ Cấu hình biến tiềm ẩn (Likert)"):
    n_vars = st.slider("Số lượng biến tiềm ẩn", 2, 10, 3)
    var_names, items_per_var = [], []
    for i in range(n_vars):
        col1, col2 = st.columns([2, 1])
        name = col1.text_input(f"Tên biến {i+1}", value=f"Var{i+1}", key=f"name_{i}")
        items = col2.number_input(f"Số câu hỏi (items)", 1, 10, 3, key=f"items_{i}")
        var_names.append(name)
        items_per_var.append(items)

with st.expander("2️⃣ Ma trận tương quan giữa các biến"):
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
                val = cols[j].number_input(f"r({var_names[i]}, {var_names[j]})", -1.0, 1.0, 0.4, key=f"r_{i}_{j}")
                row.append(val)
        cor_matrix.append(row)

with st.expander("3️⃣ Mô hình hồi quy"):
    y_var = st.selectbox("Chọn biến phụ thuộc (Y)", var_names)
    
    st.markdown("Bạn có thể tự nhập hoặc nhấn `🎲 Gợi ý tự động` để sinh hệ số hợp lý.")
    beta_input_mode = st.radio("Chế độ nhập hệ số", ["Tự nhập", "Gợi ý tự động"], horizontal=True)
    
    betas = {}
    for var in var_names:
        if var != y_var:
            if beta_input_mode == "Tự nhập":
                betas[var] = st.number_input(f"Hệ số beta: {var} → {y_var}", value=0.5, format="%.2f", key=f"manual_beta_{var}")
            else:
                # Gợi ý tự động trong khoảng 0.3–0.8, random nhẹ
                betas[var] = round(np.random.uniform(0.3, 0.8), 2)
                st.markdown(f"- `{var} → {y_var}`: **β = {betas[var]}**")


with st.expander("4️⃣ Biến định lượng"):
    n_cont_vars = st.slider("Số biến định lượng", 0, 10, 2)
    cont_config = []
    for i in range(n_cont_vars):
        st.markdown(f"**Biến định lượng {i+1}**")
        name = st.text_input(f"Tên", key=f"cont_name_{i}", value=f"ContVar{i+1}")
        mean = st.number_input(f"Trung bình (μ)", key=f"mean_{i}", value=30.0)
        std = st.number_input(f"Độ lệch chuẩn (σ)", key=f"std_{i}", value=5.0)
        round_mode = st.selectbox("Làm tròn", ["int", "float", "1", "2", "3"], index=1, key=f"round_{i}")
        round_mode = int(round_mode) if round_mode.isdigit() else round_mode
        cont_config.append({"name": name, "mean": mean, "std": std, "round": round_mode})

with st.expander("5️⃣ Biến định tính"):
    n_cat_vars = st.slider("Số biến định tính", 0, 10, 1)
    cat_config = []
    for i in range(n_cat_vars):
        st.markdown(f"**Biến định tính {i+1}**")
        name = st.text_input(f"Tên", key=f"cat_name_{i}", value=f"CatVar{i+1}")
        cat_input = st.text_input(f"Nhập danh mục (phân tách dấu phẩy)", key=f"cat_val_{i}", value="A,B")
        prob_input = st.text_input(f"Nhập xác suất (cách nhau dấu phẩy, tổng = 1)", key=f"cat_prob_{i}", value="0.6,0.4")
        try:
            categories = [x.strip() for x in cat_input.split(",")]
            probs = [float(p) for p in prob_input.split(",")]
            if len(categories) == len(probs) and abs(sum(probs) - 1.0) < 1e-6:
                cat_config.append({"name": name, "categories": categories, "probs": probs})
            else:
                st.error("⚠️ Số danh mục và xác suất không khớp hoặc tổng xác suất ≠ 1.0")
        except:
            st.error("❌ Lỗi định dạng xác suất.")

# ======== SINH DỮ LIỆU ========
st.markdown("---")
st.subheader("🚀 Sinh dữ liệu")
col1, col2 = st.columns(2)
n_samples = col1.slider("Số mẫu khảo sát", 100, 2000, 500)
likert_scale = col2.selectbox("Thang đo Likert", [5, 7], index=0)

if st.button("✨ Tạo dữ liệu"):
    try:
        df_likert = generate_likert_data_with_regression(
            var_names, items_per_var, cor_matrix, y_var, betas, n_samples, likert_scale
        )
        df_cont = generate_continuous_vars(cont_config, n_samples)
        df_cat = generate_categorical_vars(cat_config, n_samples)
        df = pd.concat([df_likert, df_cont, df_cat], axis=1)

        st.success("✅ Dữ liệu đã được tạo!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Tải xuống CSV", csv, "survey_data.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
st.markdown("---")
st.markdown("### ☕ Support me")
st.markdown("If you find this tool useful, consider buying me a coffee!")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("5DBDA424-6E66-4E58-AFD7-63836432A1C2_1_201_a.jpeg", width=120, caption="Scan to donate ☕")

with col2:
    st.markdown("""
    **📩 Contact:** hoaianthai345@gmail.com  
    **🏦 BIDV:** 6150845123 – Thái Hoài An  
    """)
    
st.markdown("Made with ❤️ by An Hoài Thái – UEH student")