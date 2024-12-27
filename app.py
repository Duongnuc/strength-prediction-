
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load CSV files
try:
    phatthai_df = pd.read_csv('phatthai.csv')
    giathanh_df = pd.read_csv('giathanh.csv')
    klr_df = pd.read_csv('klr.csv')
except FileNotFoundError as e:
    st.error(f"Lỗi: Không tìm thấy tệp dữ liệu. Đảm bảo các tệp 'phatthai.csv', 'giathanh.csv', và 'klr.csv' có trong thư mục: {e}")
    st.stop()

# Load model
try:
    model = joblib.load('model_gbm.pkl')
    if model is None:
        st.error("Mô hình không được tải thành công. Đảm bảo file 'lightgbm_model.pkl' tồn tại.")
        st.stop()
    if not callable(getattr(model, "predict", None)):
        st.error("Mô hình được tải nhưng không hợp lệ (không có hàm predict). Kiểm tra mô hình của bạn.")
        st.stop()
except FileNotFoundError as e:
    st.error(f"Lỗi: Không tìm thấy tệp mô hình 'lightgbm_model.pkl'. {e}")
    st.stop()
except Exception as e:
    st.error(f"Lỗi trong quá trình tải mô hình: {e}")
    st.stop()

# Convert dataframes to dictionaries
try:
    phatthai = phatthai_df.iloc[0].to_dict()
    giathanh = giathanh_df.iloc[0].to_dict()
    klr = klr_df.iloc[0].to_dict()
except Exception as e:
    st.error(f"Lỗi trong quá trình xử lý dữ liệu CSV: {e}")
    st.stop()

# Constants for validation
LIMITS = {
    'cement': (129, 486),
    'slag': (0, 350),
    'ash': (0, 358),
    'water': (105, 240),
    'superplastic': (0, 23),
    'coarseagg': (708, 1232),
    'fineagg': (555, 971),
}

DISPLAY_NAMES = {
    'cement': "Hàm lượng xi măng (kg/m³)",
    'slag': "Hàm lượng xỉ (kg/m³)",
    'ash': "Hàm lượng tro bay (kg/m³)",
    'water': "Hàm lượng nước (kg/m³)",
    'superplastic': "Hàm lượng siêu dẻ (kg/m³)",
    'coarseagg': "Hàm lượng cốt liệu lớn (kg/m³)",
    'fineagg': "Hàm lượng cốt liệu nhỏ (kg/m³)",
}

# Functions
def kiem_tra_cap_phoi(quy_doi_materials):
    for mat, (min_val, max_val) in LIMITS.items():
        if not (min_val <= quy_doi_materials[mat] <= max_val):
            return False, f"{DISPLAY_NAMES[mat]} không nằm trong khoảng [{min_val}, {max_val}]"
    slag_ash_ratio = (quy_doi_materials['slag'] + quy_doi_materials['ash']) / quy_doi_materials['cement']
    if not (0.3 <= slag_ash_ratio <= 0.6):
        return False, f"Tỷ lệ (Xỉ + Tro bay) / Xi măng = {slag_ash_ratio:.2f} không nằm trong khoảng [0.3, 0.6]"
    return True, "Cấp phối phù hợp"

def quy_doi_ve_1m3(materials, klr):
    total_volume = sum(materials[mat] / klr[mat] for mat in materials.keys())
    he_so_quy_doi = 1000 / total_volume
    return {mat: materials[mat] * he_so_quy_doi for mat in materials.keys()}

def du_doan_cuong_do(quy_doi_materials, tuoi_list):
    predictions = []
    for tuoi in tuoi_list:
        inputs = [
            quy_doi_materials['cement'],
            quy_doi_materials['slag'],
            quy_doi_materials['ash'],
            quy_doi_materials['water'],
            quy_doi_materials['superplastic'],
            quy_doi_materials['coarseagg'],
            quy_doi_materials['fineagg'],
            tuoi
        ]
        # Validate and reshape inputs
        try:
            inputs = np.array(inputs, dtype=np.float64).reshape(1, -1)
        except ValueError as e:
            raise ValueError(f"Invalid inputs for prediction: {inputs}. Error: {e}")
        try:
            st.write("Dữ liệu đầu vào cho dự đoán:", inputs)
            if model is not None:
                predictions.append(model.predict(inputs)[0])
            else:
                raise ValueError("Model is not properly loaded or initialized.")
        except Exception as e:
            st.error(f"Lỗi khi dự đoán với đầu vào: {inputs}. Chi tiết lỗi: {e}")
            raise e
    return predictions

def tinh_gia_thanh_va_phat_thai(quy_doi_materials, giathanh, phatthai, predictions):
    tong_gia_thanh = sum(quy_doi_materials[mat] * giathanh[mat] for mat in quy_doi_materials)
    tong_phat_thai = sum(quy_doi_materials[mat] * phatthai[mat] for mat in quy_doi_materials)
    gia_thanh_mpa = tong_gia_thanh / predictions[-1]
    phat_thai_mpa = tong_phat_thai / predictions[-1]
    return tong_gia_thanh, tong_phat_thai, gia_thanh_mpa, phat_thai_mpa

def ve_duong_xu_huong(tuoi_list, predictions):
    def logistic_growth(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))
    try:
        params, _ = curve_fit(logistic_growth, tuoi_list, predictions, maxfev=10000, bounds=(0, [np.inf, np.inf, np.inf]))
        tuoi_fit = np.linspace(min(tuoi_list), max(tuoi_list), 100)
        predictions_fit = logistic_growth(tuoi_fit, *params)
        plt.figure(figsize=(8, 5))
        plt.scatter(tuoi_list, predictions, color='blue', label="Cường độ thực tế")
        plt.plot(tuoi_fit, predictions_fit, color='red', linestyle='--', label="Đường xu hướng")
        plt.title("Cường độ bê tông theo ngày tuổi")
        plt.xlabel("Ngày tuổi")
        plt.ylabel("Cường độ (MPa)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Lỗi khi vẽ đồ thị: {e}")

# Streamlit UI
st.title("Dự đoán cường độ bê tông và tính toán kinh tế, phát thải")

st.header("Nhập liệu hàm lượng vật liệu (kg)")
materials = {
    'cement': st.number_input("Xi măng (kg):", value=350.0),
    'slag': st.number_input("Xỉ (kg):", value=100.0),
    'ash': st.number_input("Tro bay (kg):", value=80.0),
    'water': st.number_input("Nước (kg):", value=150.0),
    'superplastic': st.number_input("Siêu dẻ (kg):", value=5.0),
    'coarseagg': st.number_input("Cốt liệu lớn (kg):", value=900.0),
    'fineagg': st.number_input("Cốt liệu nhỏ (kg):", value=800.0),
}

if st.button("Quy đổi về 1m³, dự đoán và tính toán"):
    try:
        quy_doi_materials = quy_doi_ve_1m3(materials, klr)
        is_valid, message = kiem_tra_cap_phoi(quy_doi_materials)
        st.subheader("Kết quả kiểm tra cấp phối:")
        st.write(message)

        if is_valid:
            tuoi_list = [3, 7, 28, 91]
            predictions = du_doan_cuong_do(quy_doi_materials, tuoi_list)
            try:
                tong_gia_thanh, tong_phat_thai, gia_thanh_mpa, phat_thai_mpa = tinh_gia_thanh_va_phat_thai(
                    quy_doi_materials, giathanh, phatthai, predictions)
                
                st.subheader("Cấp phối đã quy đổi về 1m³:")
                for mat, value in quy_doi_materials.items():
                    st.write(f"{DISPLAY_NAMES[mat]}: {value:.2f} kg")

                st.subheader("Kết quả kinh tế và phát thải:")
                st.markdown(f'Tổng giá thành: <b style="color:red;">{tong_gia_thanh:.2f} VNĐ</b>', unsafe_allow_html=True)
                st.markdown(f'Lượng phát thải: <b style="color:red;">{tong_phat_thai:.2f} kg</b>', unsafe_allow_html=True)
                st.markdown(f'Giá thành/MPa: <b style="color:red;">{gia_thanh_mpa:.2f} VNĐ/MPa</b>', unsafe_allow_html=True)
                st.markdown(f'CO2/MPa: <b style="color:red;">{phat_thai_mpa:.2f} kg CO2/MPa</b>', unsafe_allow_html=True)

                ve_duong_xu_huong(tuoi_list, predictions)
            except Exception as e:
                st.error(f"Lỗi trong quá trình tính toán kinh tế và phát thải: {e}")
        else:
            st.warning("Không thể tiến hành dự đoán do cấp phối không phù hợp.")
    except Exception as e:
        st.error(f"Lỗi tổng quát: {e}")
