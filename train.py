import torch
import os
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 필요한 크기로 조정
    transforms.ToTensor()
])

# TensorBoard 설정
writer = SummaryWriter(log_dir=os.path.join(params.model_save_path, "logs"))

# 데이터 로더 설정
train_dataset = CustomDataset(root_dir=params.train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

# 모델 초기화
cnn_lstm_model = CNNLSTM(params.input_channels, params.num_classes, params.hidden_dim, params.num_layers).to(params.device)

# 옵티마이저 및 손실 함수 정의
optimizer_cnn = torch.optim.Adam(cnn_lstm_model.parameters(), lr=params.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

print("train.py is running...")

# 모델 저장 디렉토리 생성
if not os.path.exists(params.model_save_path):
    os.makedirs(params.model_save_path)

# 학습 루프 정의
def train_model(model, optimizer, criterion, train_loader, epochs):
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}] started...")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # tqdm을 사용해 진행 상황을 표시
        for data, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            data, labels = data.to(params.device), labels.to(params.device)

            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf found in loss, skipping this batch.")
                continue
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # TensorBoard에 손실과 정확도 기록
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# CNN-LSTM 모델 학습 실행
def train_cnn_lstm():
    print("Training CNN-LSTM model started...")
    train_model(cnn_lstm_model, optimizer_cnn, criterion, train_loader, params.epochs)
    print("Training completed. Saving the model...")
    torch.save(cnn_lstm_model.state_dict(), params.cnn_lstm_model_file)
    print(f"Model saved at {params.cnn_lstm_model_file}")

if __name__ == "__main__":
    print("Starting the training script...")
    train_cnn_lstm()

# TensorBoard 종료
writer.close()
