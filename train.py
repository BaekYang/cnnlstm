import torch
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 필요한 크기로 조정
    transforms.ToTensor()
])

# 데이터 로더 설정
train_dataset = CustomDataset(root_dir=params.train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

# 모델 초기화
cnn_lstm_model = CNNLSTM(params.input_channels, params.num_classes, params.hidden_dim, params.num_layers).to(params.device)

# 옵티마이저 및 손실 함수 정의
optimizer_cnn = torch.optim.Adam(cnn_lstm_model.parameters(), lr=params.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 학습 루프 정의
def train_model(model, optimizer, criterion, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(params.device), labels.to(params.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# CNN-LSTM 모델 학습 실행
def train_cnn_lstm():
    print("Training CNN-LSTM model...")
    train_model(cnn_lstm_model, optimizer_cnn, criterion, train_loader, params.epochs)
    torch.save(cnn_lstm_model.state_dict(), params.cnn_lstm_model_file)
