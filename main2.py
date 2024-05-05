import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np


# 啟動網路攝影機

camera = cv2.VideoCapture(0)

# use_camera = st.checkbox("使用網絡攝像頭")

# if use_camera:
#     camera = cv2.VideoCapture(0)  # 0表示默認相機
# else:
#     camera = None

# while True:
#     if use_camera:
#         _, frame = camera.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame)
#     else:
#         break


# 載入YOLOv8模型
model = YOLO("./smoking_Yolov8.pt")

# Streamlit應用程式標題
st.title("YOLOv8 實時目標檢測")


# 初始化影像
img = None

# 拍照按鈕
take_photo = st.button("拍照")

# 如果按下拍照按鈕
if take_photo:
    # 從相機讀取影像
    ret, img = camera.read()

    # 如果成功讀取影像
    if ret:
        # 將OpenCV格式轉換為PIL格式
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 使用YOLOv8模型進行目標檢測
        results = model.predict(image)

        # 在原始影像上繪製檢測結果
        annotated_img = results[0].plot()
        annotated_img = np.asarray(annotated_img)

        # 顯示帶有目標檢測框的影像
        st.image(annotated_img, channels="BGR")

# 釋放相機資源
camera.release()