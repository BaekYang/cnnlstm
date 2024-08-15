import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim, num_layers):
        super(CNNLSTM, self).__init__()
        
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # CNN Encoder
        batch_size, C, H, W = x.size()
        c_in = x.view(batch_size, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.view(batch_size, -1)
        
        # LSTM Decoder
        lstm_out, _ = self.lstm(c_out)
        
        # Fully Connected Layer
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out
