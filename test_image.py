import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from emotion_model import EfficientEmotion  # 사용자 정의 모델
import matplotlib.pyplot as plt

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 라벨
emotion_labels = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger']

# MTCNN 얼굴 검출기
mtcnn = MTCNN(keep_all=True, device=device)

# 모델 불러오기
model = EfficientEmotion().to(device)
model.load_state_dict(torch.load("model\\best_ferplus_emotion_model_efficient_surprise_focus.pth", map_location=device))
model.eval()

# 이미지 경로
#image_path = "testImage/surprise.jpeg"
#image_path = "testImage/surprise1.jpeg"
#image_path = "testImage/surprise2.jpg"
#image_path = "testImage/surprise3.jpg"
image_path = "testImage/surprise4.jpg"
#image_path = "testImage/happy1.jpg"
#image_path = "testImage/neutral.jpg"
#image_path = "testImage/neutral1.jpg"
#image_path = "testImage/sad1.png"
#image_path = "testImage/sad2.png"
#image_path = "testImage\\angry1.jpg"
#image_path = "testImage\\angry2.jpg"
img_pil = Image.open(image_path).convert("RGB")
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # OpenCV용 BGR

# 얼굴 검출
boxes, _ = mtcnn.detect(img_pil)

if boxes is None:
    print("❌ 얼굴을 찾을 수 없습니다.")
else:
    print(f"✅ 총 {len(boxes)}개의 얼굴을 찾았습니다.\n")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 얼굴 crop (PIL 방식)
        face = img_pil.crop((x1, y1, x2, y2))

        # 전처리
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)

        # 예측
        with torch.no_grad():
            output = model(face_tensor)
            probs = F.softmax(output, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            top_emotion = emotion_labels[top_idx]
            top_prob = probs[top_idx].item()

        # 로그 출력
        print(f"[얼굴 {i+1}] 위치: ({x1},{y1}) ~ ({x2},{y2})")
        print(f"예측 감정: {top_emotion} ({top_prob:.2f})")
        for idx, prob in enumerate(probs):
            print(f"  - {emotion_labels[idx]}: {prob.item():.4f}")
        print()

        # 시각화 - 이미지에 박스와 텍스트 추가
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{top_emotion} ({top_prob:.2f})"
        cv2.putText(img_cv, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 저장 및 표시
    output_path = "output.jpg"
    cv2.imwrite(output_path, img_cv)
    print(f"🖼️ 결과 이미지 저장됨: {output_path}")

    # plt로 표시
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title("Prediction Result")
    plt.axis("off")
    plt.show()
