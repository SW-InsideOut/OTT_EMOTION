
# ✅ 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os

# ✅ 하이퍼파라미터 설정
IMG_WIDTH, IMG_HEIGHT = 48, 48  # 입력 이미지 크기
BATCH_SIZE = 32
EPOCHS = 100  # 학습 횟수
NUM_CLASSES = 5  # 클래스 개수
DATASET_PATH = "C://Users//imyy1//Downloads//train-20250416T040127Z-001//train"  # 데이터 경로 (사용자 지정)

# ✅ 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",  # 흑백 이미지라면 grayscale
    class_mode="categorical",
    subset="training"
)
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)


# ✅ CNN 모델 정의
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# ✅ 모델 컴파일
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    filepath="best_model_local11.h5",   # 저장할 모델 경로
    monitor="val_accuracy",            # 모니터할 값
    save_best_only=True,               # 가장 좋은 결과만 저장
    mode="max",                        # val_accuracy가 클수록 좋음
    verbose=1                          # 저장될 때마다 로그 출력
)

# ✅ 모델 학습
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS,callbacks=[checkpoint]  # 콜백 적용
                    )


# ✅ 모델 저장
#model.save("best_model_local5.h5")

# ✅ 학습 이력 저장
with open("training_history_local11.pkl", "wb") as f:
    pickle.dump(history.history, f)

"""
# ✅ 클래스 리스트 확인
class_labels = list(train_generator.class_indices.keys())
print("클래스 리스트:", class_labels)
"""