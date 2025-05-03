"""
import cv2
import numpy as np
import tensorflow as tf
import time

# 🎯 Haar Cascade 로드 (정면 얼굴 검출)
face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

# 🎯 감정 분석 모델 로드
MODEL_PATH = "../models/best_model_local8.h5"  # 실제 모델 파일 경로를 지정하세요
model = tf.keras.models.load_model(MODEL_PATH)

# 감정 클래스 라벨 (순서가 중요함!)
class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprize']

# 🎥 웹캠 열기 (0번: 기본 카메라)
cap = cv2.VideoCapture(0)

# 🎯 상태 관리 변수
current_emotion = "neutral"  # 현재 표시할 감정
previous_emotion = "neutral"  # 직전 감정
emotion_change_time = time.time()  # 감정 바뀐 시간 기록

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0
        face_reshaped = np.expand_dims(face_resized, axis=(0, -1))

        predictions = model.predict(face_reshaped, verbose=0)
        predicted_class = np.argmax(predictions)
        new_emotion = class_labels[predicted_class]

        now = time.time()

        # ✅ neutral ➡️ 다른 감정: 바로 반영
        if previous_emotion == "neutral" and new_emotion != "neutral":
            current_emotion = new_emotion
            emotion_change_time = now
            previous_emotion = new_emotion  # 👉 감정 전환 시에만 업데이트
            print(f"🔄 Neutral ➡️ {new_emotion}")

        # ✅ 다른 감정 ➡️ 다른 감정: 3초 유지
        elif previous_emotion != "neutral" and new_emotion != previous_emotion:
            if now - emotion_change_time > 3:
                current_emotion = new_emotion
                emotion_change_time = now
                previous_emotion = new_emotion  # 👉 감정 전환 시에만 업데이트
                print(f"⏱️ 감정 유지 후 변경: {previous_emotion} ➡️ {new_emotion}")

        # ✅ 감정이 유지되거나 neutral 감정일 때: 업데이트 (previous_emotion 유지!)
        elif new_emotion == previous_emotion or new_emotion == "neutral":
            current_emotion = new_emotion
            emotion_change_time = now
            # ❌ previous_emotion 은 여기서 업데이트하지 않음

        # 🔹 얼굴 및 감정 라벨 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import tensorflow as tf
import time

# 🎯 Haar Cascade 로드 (정면 얼굴 검출)
face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

# 🎯 감정 분석 모델 로드
MODEL_PATH = "../models/best_model_local8.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 감정 클래스 라벨
class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprize']

# 🎥 웹캠 열기
cap = cv2.VideoCapture(0)

# 🎯 상태 관리 변수
current_emotion = "neutral"
previous_emotion = "neutral"
emotion_change_time = time.time()

while cap.isOpened():
    frame_start = time.time()  # 📸 프레임 시작 시간

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0
        face_reshaped = np.expand_dims(face_resized, axis=(0, -1))

        # ⏱️ 감정 예측 시간 측정
        pred_start = time.time()
        predictions = model.predict(face_reshaped, verbose=0)
        pred_end = time.time()
        print(f"⏱️ 감정 분류 소요 시간: {pred_end - pred_start:.4f}초")

        predicted_class = np.argmax(predictions)
        new_emotion = class_labels[predicted_class]

        now = time.time()

        # 🔄 감정 상태 전환 로직
        if previous_emotion == "neutral" and new_emotion != "neutral":
            current_emotion = new_emotion
            emotion_change_time = now
            previous_emotion = new_emotion
            print(f"🔄 Neutral ➡️ {new_emotion}")

        elif previous_emotion != "neutral" and new_emotion != previous_emotion:
            if now - emotion_change_time > 3:
                current_emotion = new_emotion
                emotion_change_time = now
                previous_emotion = new_emotion
                print(f"⏱️ 감정 유지 후 변경: {previous_emotion} ➡️ {new_emotion}")

        elif new_emotion == previous_emotion or new_emotion == "neutral":
            current_emotion = new_emotion
            emotion_change_time = now

        # 얼굴 및 감정 라벨 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Real-time Emotion Detection", frame)

    # 📸 FPS 측정
    frame_end = time.time()
    fps = 1.0 / (frame_end - frame_start)
    print(f"📸 프레임 처리 속도: {fps:.2f} FPS")

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
