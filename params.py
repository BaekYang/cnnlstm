import torch

# 데이터 경로 설정
train_data_path = "video"
test_data_path = "test"
validation_data_path = "validation"

# 학습 파라미터
batch_size = 32
learning_rate = 0.001
epochs = 50

# 모델 파라미터
input_channels = 3  # 예: RGB 이미지
hidden_dim = 256
num_layers = 2
num_classes = 3  # 3개의 클래스: 0-standing, 1-ready, 2-aiming

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 저장 경로
model_save_path = "models/"
cnn_lstm_model_file = model_save_path + "cnn_lstm_model.pth"
