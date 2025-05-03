import numpy as np
import tensorflow as tf
import cv2
import os

# ✅ 모델 불러오기
MODEL_PATH = "../models/best_model_local8.h5"  # 저장한 모델 경로
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ 클래스 이름 (학습할 때 사용한 클래스 순서와 동일해야 함)
class_labels = ["Angry", "Happy", "neutral","Sad", "Surprise"]  # 예시 (수정 가능)
# ✅ 얼굴 검출기 (Haar Cascade 사용)
face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
profile_cascade=cv2.CascadeClassifier("../haarcascade_profileface.xml")

# ✅ 테스트할 이미지 경로 (변경 가능)
#TEST_IMAGE_PATH = "C:\\Users\\imyy1\\Downloads\\charles-etoroma-95UF6LXe-Lo-unsplash.jpg"
#TEST_IMAGE_PATH = "C:\\Users\\imyy1\\Downloads\\1464523904287.jpg"
#화남

#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\istockphoto-176811710-612x612.jpg"
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\VD52317198_w640.jpg"
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\KlppBCGbqo.jpeg"
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\IMG_4128.jpeg"
#해피
TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\gettyimages-jv11208462.jpg"
#슬픔
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\images(2).jpg"
#무표정인데 슬픔으로 인식
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\PS18041100122.jpg"
#TEST_IMAGE_PATH ="C:\\Users\\imyy1\\Downloads\\202211171437273510_1.webp"


"""
# ✅ 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지 로드
    img = cv2.resize(img, (48, 48))  # 모델 입력 크기에 맞게 조정
    img = img.astype("float32") / 255.0  # 정규화
    img = np.expand_dims(img, axis=-1)  # 채널 차원 추가 (48, 48, 1)
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가 (1, 48, 48, 1)
    return img

# ✅ 이미지 불러오기 및 전처리
processed_image = preprocess_image(TEST_IMAGE_PATH)

# ✅ 예측 수행
predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions)  # 가장 높은 확률의 클래스 선택
# ✅ 확률을 백분율로 변환
percentage_probs = predictions * 100
# ✅ 결과 출력
print(f"예측 결과: {class_labels[predicted_class]}")
print("클래스별 확률 (백분율):")
for i, prob in enumerate(percentage_probs[0]):
    print(f"  {class_labels[i]}: {prob:.2f}%")
"""
def iou(box1, box2):
    """ 두 박스 간의 IoU (교집합/합집합) 계산 """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def remove_duplicates(faces, iou_threshold=0.3):
    """ IoU 기반 중복 얼굴 제거 """
    final_faces = []
    for face in faces:
        if all(iou(face, f) < iou_threshold for f in final_faces):
            final_faces.append(face)
    return final_faces

def detect_face_and_predict(image_path):
    """ 이미지에서 얼굴을 검출하고 감정 예측을 수행하는 함수 """

    # ✅ 이미지 로드 및 Grayscale 변환
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ 정면 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))

    # ✅ 왼쪽 측면 얼굴 검출
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # ✅ 오른쪽 측면 얼굴 검출 (좌우 반전 후)
    flipped_gray = cv2.flip(gray, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # ✅ 오른쪽 측면 얼굴 좌표 변환
    img_width = gray.shape[1]
    flipped_profiles_converted = [(img_width - x - w, y, w, h) for (x, y, w, h) in flipped_profiles]

    # ✅ 모든 얼굴 좌표 합치기 (빈 리스트 제거 후 합침)
    all_faces = []
    if len(faces) > 0:
        all_faces.extend(faces.tolist())
    if len(profiles) > 0:
        all_faces.extend(profiles.tolist())
    if len(flipped_profiles_converted) > 0:
        all_faces.extend(flipped_profiles_converted)

    # ✅ 중복 제거 (IoU 기준)
    final_faces = remove_duplicates(all_faces)

    # ✅ 얼굴이 감지되지 않았을 경우 처리
    if len(final_faces) == 0:
        print("🚨 얼굴을 찾을 수 없습니다.")
        return
    # ✅ 얼굴 감정 분석
    print(f"📸 감지된 얼굴 개수: {len(final_faces)}")  # 중복 제거 후 얼굴 개수 확인

    # ✅ 얼굴 감정 분석
    for (x, y, w, h) in final_faces:
        # 얼굴 크롭 및 전처리
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  
        face_normalized = face_resized / 255.0  
        face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))  

        # ✅ 감정 예측
        predictions = model.predict(face_reshaped)
        predicted_class = np.argmax(predictions)
        percentages = (predictions[0] * 100).round(2)  # 백분율 변환

        # ✅ 결과 출력
        print(f"👤 감정 분석 결과: {class_labels[predicted_class]}")
        print("📊 클래스별 확률 (%):")
        for label, percentage in zip(class_labels, percentages):
            print(f"  {label}: {percentage:.2f}%")

        # ✅ 얼굴 영역 표시
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, class_labels[predicted_class], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ✅ 결과 이미지 출력
    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"✅ 모델 출력층 노드 수: {model.output_shape}")

# ✅ 실행
detect_face_and_predict(TEST_IMAGE_PATH)