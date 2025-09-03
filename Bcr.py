import streamlit as st
import pandas as pd
import random
import plotly.express as px
from collections import Counter, defaultdict
import os

# --- Cấu hình trang ---
st.set_page_config(page_title="🎯 Baccarat Predictor Pro", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🎯 Baccarat Predictor Pro")
    raw_input = st.text_area("🔢 Nhập chuỗi kết quả (P, B, T):", height=150)
    st.markdown("📌 Ví dụ: `P,B,P,T,B,P,P,B,T,P`")
    model_option = st.selectbox("🧠 Chọn mô hình dự đoán:", ["Thống kê", "Markov", "Bayesian", "Deep Learning"])
    predict_button = st.button("🔮 Dự đoán tiếp theo")
    st.markdown("---")
    user_note = st.text_area("📝 Ghi chú cá nhân", height=100)
    save_button = st.button("💾 Lưu lịch sử & ghi chú")

# --- Xử lý dữ liệu ---
def parse_input(data):
    return [x.strip().upper() for x in data.split(",") if x.strip().upper() in ["P", "B", "T"]]

def count_streaks(data):
    if not data: return []
    streaks = []
    current = data[0]
    count = 1
    for x in data[1:]:
        if x == current:
            count += 1
        else:
            streaks.append((current, count))
            current = x
            count = 1
    streaks.append((current, count))
    return streaks

def smart_predict(data):
    counts = Counter(data[-10:])
    if counts["P"] > counts["B"]:
        return "Player"
    elif counts["B"] > counts["P"]:
        return "Banker"
    return random.choice(["Player", "Banker", "Tie"])

def markov_predict(data):
    if len(data) < 2: return "Không đủ dữ liệu"
    transitions = {"P": {"P":0, "B":0, "T":0}, "B": {"P":0, "B":0, "T":0}, "T": {"P":0, "B":0, "T":0}}
    for i in range(len(data)-1):
        transitions[data[i]][data[i+1]] += 1
    last = data[-1]
    next_probs = transitions[last]
    return max(next_probs, key=next_probs.get)

def bayesian_predict(data, window=2):
    if len(data) < window + 1: return "Không đủ dữ liệu"
    patterns = defaultdict(lambda: {"P":0, "B":0, "T":0})
    for i in range(len(data) - window):
        key = tuple(data[i:i+window])
        next_ = data[i+window]
        patterns[key][next_] += 1
    last_pattern = tuple(data[-window:])
    if last_pattern not in patterns: return "Không có mẫu phù hợp"
    next_probs = patterns[last_pattern]
    return max(next_probs, key=next_probs.get)

def deep_learning_predict(data):
    if len(data) < 20:
        return "Cần ít nhất 20 kết quả để mô hình AI hoạt động"
    simulated = random.choices(["Player", "Banker", "Tie"], weights=[0.4, 0.4, 0.2])[0]
    return f"Dự đoán AI (giả lập): {simulated}"

# --- Tự động cập nhật từ file CSV ---
def load_real_data(file_path="du_lieu_thuc.csv"):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df["Kết quả"].tolist()
    return []

# --- Phân tích dữ liệu ---
parsed_data = parse_input(raw_input)
st.title("📊 Phân Tích & Dự Đoán Baccarat")

if raw_input and not parsed_data:
    st.warning("⚠️ Dữ liệu không hợp lệ. Vui lòng nhập chuỗi gồm các ký tự P, B, T ngăn cách bằng dấu phẩy.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📜 Lịch sử & Thống kê")
    st.write(f"Tổng số ván: {len(parsed_data)}")
    st.write(parsed_data)

    if parsed_data:
        df_counts = pd.DataFrame(Counter(parsed_data).items(), columns=["Kết quả", "Số lần"])
        fig = px.bar(df_counts, x="Kết quả", y="Số lần", color="Kết quả", title="📊 Tỷ lệ kết quả")
        st.plotly_chart(fig, use_container_width=True)

        streaks = count_streaks(parsed_data)
        df_streaks = pd.DataFrame(streaks, columns=["Kết quả", "Chuỗi"])
        fig2 = px.line(df_streaks, y="Chuỗi", title="📈 Chuỗi liên tiếp")
        st.plotly_chart(fig2, use_container_width=True)

    real_data = load_real_data()
    if real_data:
        st.subheader("📂 Dữ liệu thực tế từ file")
        st.write(f"Số ván: {len(real_data)}")
        df_real = pd.DataFrame(Counter(real_data).items(), columns=["Kết quả", "Số lần"])
        fig_real = px.pie(df_real, names="Kết quả", values="Số lần", title="📊 Tỷ lệ kết quả thực tế")
        st.plotly_chart(fig_real, use_container_width=True)

with col2:
    st.subheader("🔮 Dự đoán tiếp theo")
    if predict_button and parsed_data:
        if model_option == "Thống kê":
            prediction = smart_predict(parsed_data)
        elif model_option == "Markov":
            prediction = markov_predict(parsed_data)
        elif model_option == "Bayesian":
            prediction = bayesian_predict(parsed_data)
        elif model_option == "Deep Learning":
            prediction = deep_learning_predict(parsed_data)
        st.success(f"✅ Dự đoán: **{prediction}**")
    else:
        st.info("⏳ Nhấn nút 'Dự đoán tiếp theo' để xem kết quả")

    st.subheader("📝 Ghi chú của bạn")
    if raw_input and user_note:
        st.code(user_note)
    elif raw_input:
        st.info("Bạn có thể nhập ghi chú bên sidebar")

# --- Lưu lịch sử & ghi chú ---
if save_button and parsed_data:
    df_history = pd.DataFrame({"Kết quả": parsed_data})
    df_history.to_csv("lich_su_baccarat.csv", index=False)
    with open("ghi_chu.txt", "a", encoding="utf-8") as f:
        f.write(f"{raw_input} → {user_note}\n")
    st.success("📁 Lịch sử & ghi chú đã được lưu")

st.markdown("---")
st.caption("© 2025 Baccarat Predictor Pro — Powered by Az & Copilot")
