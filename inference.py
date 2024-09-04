import torch
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms

from sklearn.metrics import f1_score
# 평가 함수 정의
def evaluate_model(model, data_loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(params.device), labels.to(params.device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf found in loss, skipping this batch.")
                continue
            
            total_loss += loss.item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    f1 = 100. * f1_score(all_labels, all_preds, average="weighted")
    return accuracy, avg_loss, f1
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
    cnn_lstm_accuracy, loss, f1 = evaluate_model(cnn_lstm_model, test_loader)
    print(f"CNN-LSTM Accuracy: {cnn_lstm_accuracy * 100:.2f}%")
    print(f"CNN-LSTM Loss: {loss:.4f}")
    print(f"CNN-LSTM F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
    