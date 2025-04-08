import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Function to generate synthetic Likert data with target Cronbach's Alpha
def generate_likert_data(n_samples, n_items, target_alpha):
    # Approximate average inter-item correlation from target alpha
    r = target_alpha / (n_items - target_alpha * (n_items - 1))
    
    # Create correlation matrix
    corr_matrix = np.full((n_items, n_items), r)
    np.fill_diagonal(corr_matrix, 1)

    # Convert correlation matrix to covariance matrix (std dev = 1)
    cov_matrix = corr_matrix

    # Generate multivariate normal data
    mv_data = multivariate_normal.rvs(mean=[0]*n_items, cov=cov_matrix, size=n_samples)

    # Convert to Likert scale (1 to 5)
    likert_data = np.clip(np.round(np.interp(mv_data, (mv_data.min(), mv_data.max()), (1, 5))), 1, 5)
    likert_df = pd.DataFrame(likert_data, columns=[f"item_{i+1}" for i in range(n_items)])

    return likert_df

# Streamlit UI
st.title("🔢 Fake Likert Data Generator for Target Cronbach's Alpha")

n_samples = st.number_input("Số mẫu (n):", min_value=10, max_value=10000, value=100)
n_items = st.number_input("Số câu hỏi (item):", min_value=2, max_value=50, value=5)
target_alpha = st.slider("Cronbach's Alpha mong muốn:", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if st.button("Tạo dữ liệu"):
    df = generate_likert_data(n_samples, n_items, target_alpha)
    st.success(f"Đã tạo dữ liệu với {n_samples} mẫu và {n_items} biến.")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Tải dữ liệu CSV", data=csv, file_name="likert_data.csv", mime="text/csv")
