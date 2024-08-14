import train
import inference

# 모델 학습
train.train_cnn_lstm()

# 모델 평가
inference.main()
