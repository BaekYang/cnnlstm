import torch
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms

# 평가 함수 정의
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(params.device), labels.to(params.device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
def main():
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델에 맞는 크기로 조정
        transforms.ToTensor()
    ])

    # 테스트 데이터 로드
    test_dataset = CustomDataset(root_dir=params.test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # 데이터 확인
    print(f"Test Dataset: {len(test_dataset)} samples")
    
    # 모델 로드
    cnn_lstm_model = CNNLSTM(params.input_channels, params.num_classes, params.hidden_dim, params.num_layers).to(params.device)
    cnn_lstm_model.load_state_dict(torch.load(params.cnn_lstm_model_file))



    # CNN-LSTM 모델 평가
    cnn_lstm_accuracy = evaluate_model(cnn_lstm_model, test_loader)
    print(f"CNN-LSTM Accuracy: {cnn_lstm_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
    