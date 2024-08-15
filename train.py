import torch
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms
from inference import evaluate_model

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Handle grayscale to RGB conversion if needed
])

# 데이터 로더 설정
train_dataset = CustomDataset(root_dir=params.train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

validation_dataset = CustomDataset(root_dir=params.validation_data_path, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=params.batch_size, shuffle=False)

# 데이터 확인
print(f"Train Dataset: {len(train_dataset)} samples")
print(f"Validation Dataset: {len(validation_dataset)} samples")

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
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Accuracy: {evaluate_model(model, validation_loader) * 100:.2f}%")

# CNN-LSTM 모델 학습 실행
def train_cnn_lstm():

    print("Training CNN-LSTM model...")
    train_model(cnn_lstm_model, optimizer_cnn, criterion, train_loader, params.epochs)
    torch.save(cnn_lstm_model.state_dict(), params.cnn_lstm_model_file)
