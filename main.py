import train
import inference
import params

print(f"using device: {params.device}")
# 모델 학습
train.train_cnn_lstm()

# 모델 평가
inference.main()
