import os
import torch

# 현재 스크립트 파일의 위치를 기준으로 프로젝트 루트 경로를 설정합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로 설정 (상대경로)
import os
import torch

# 현재 스크립트 파일의 위치를 기준으로 프로젝트 루트 경로를 설정합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로 설정 (상대경로)
train_data_path = os.path.join(base_dir, "video")
test_data_path = os.path.join(base_dir, "test")
validation_data_path = os.path.join(base_dir, "validation")

# 학습 파라미터
batch_size = 32
learning_rate = 0.001
epochs = 1

# 모델 파라미터
input_channels = 3  # 예: RGB 이미지
hidden_dim = 256
num_layers = 2
num_classes = 3  # 3개의 클래스: 0-standing, 1-ready, 2-aiming

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 저장 경로 (상대경로)
model_save_path = os.path.join(base_dir, "models")
cnn_lstm_model_file = os.path.join(model_save_path, "cnn_lstm_model.pth")



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

# 모델 저장 경로 (상대경로)
model_save_path = os.path.join(base_dir, "models")
cnn_lstm_model_file = os.path.join(model_save_path, "cnn_lstm_model.pth")
