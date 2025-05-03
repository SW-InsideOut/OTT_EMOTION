import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# ✅ 저장된 모델 불러오기
model = load_model("../models/best_model_local10.h5")

# ✅ 저장된 학습 이력 불러오기
with open("../history/training_history_local10.pkl", "rb") as f:
    history = pickle.load(f)
# ✅ 최종 정확도 출력
final_train_acc = history['accuracy'][-1]
final_val_acc = history['val_accuracy'][-1]

print(f"✅ 최종 학습 정확도: {final_train_acc * 100:.2f}%")
print(f"✅ 최종 검증 정확도: {final_val_acc * 100:.2f}%")
# ✅ 최종 손실값 출력
final_train_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]

print(f"✅ 최종 학습 손실 (loss): {final_train_loss:.4f}")
print(f"✅ 최종 검증 손실 (val_loss): {final_val_loss:.4f}")
# ✅ 학습 정확도 및 손실 그래프 그리기
plt.figure(figsize=(12, 4))

# 학습 정확도
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# 학습 손실
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()

