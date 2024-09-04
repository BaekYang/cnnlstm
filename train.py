import torch
import os
import time
from torch.utils.data import DataLoader
import params
from model import CNNLSTM
from dataset import CustomDataset
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import inference
from sklearn.metrics import f1_score

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to required size
    transforms.ToTensor()
])

# TensorBoard setup
tensorboard_path = os.path.join(params.model_save_path, f"{time.strftime('%m-%d_%H.%M', time.localtime(time.time()))}-logs")
os.makedirs(tensorboard_path, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_path)
print(f"Writing logs in {tensorboard_path}...")

# DataLoader setup
train_dataset = CustomDataset(root_dir=params.train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
validation_dataset = CustomDataset(root_dir=params.validation_data_path, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=params.batch_size, shuffle=False)

# Model initialization
cnn_lstm_model = CNNLSTM(params.input_channels, params.num_classes, params.hidden_dim, params.num_layers).to(params.device)

# Optimizer and loss function
optimizer_cnn = torch.optim.Adam(cnn_lstm_model.parameters(), lr=params.learning_rate)
criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cnn, step_size=10, gamma=0.1)

print("train.py is running...")

# Create model saving directory
if not os.path.exists(params.model_save_path):
    os.makedirs(params.model_save_path)

# Training loop definition
def train_model(model, optimizer, criterion, train_loader, epochs, valid_loader):
    best_val_f1 = 0
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}] started...")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        f1 = 100. * f1_score(all_labels, all_preds, average="weighted")

        writer.add_scalar('Train_Loss', avg_loss, epoch)
        writer.add_scalar('Train_Accuracy', accuracy, epoch)
        writer.add_scalar('Train_F1_score', f1, epoch)
        
        validation_accuracy, validation_loss, validation_f1 = inference.evaluate_model(model, valid_loader)
        writer.add_scalar('Validation_Accuracy', validation_accuracy, epoch)
        writer.add_scalar('Validation_Loss', validation_loss, epoch)
        writer.add_scalar('Validation_F1_score', validation_f1, epoch)
        print(f"Epoch [{epoch+1}/{epochs}], T_Loss: {avg_loss:.2f}, T_Accuracy: {accuracy:.2f}%, T_F1: {f1:.2f}%\n \
              V_Loss: {validation_loss:.2f}, V_Accuracy: {validation_accuracy:.2f}%, V_F1: {validation_f1:.2f}%")
        
        # scheduler.step()

        if validation_f1 > best_val_f1:
            best_val_f1 = validation_f1
            print("Validation F1 improved, saving model...")
            torch.save(model.state_dict(), os.path.join(params.model_save_path, 'best_model.pth'))

# CNN-LSTM model training execution
def train_cnn_lstm():
    print("Training CNN-LSTM model started...")
    try:
        train_model(cnn_lstm_model, optimizer_cnn, criterion, train_loader, params.epochs, validation_loader)
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        print("Training completed. Saving the final model...")
        torch.save(cnn_lstm_model.state_dict(), params.cnn_lstm_model_file)
        print(f"Model saved at {params.cnn_lstm_model_file}")

if __name__ == "__main__":
    print("Starting the training script...")
    train_cnn_lstm()

# Close TensorBoard
writer.close()
