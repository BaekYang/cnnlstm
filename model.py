import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim, num_layers):
        super(CNNLSTM, self).__init__()

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # LSTM Decoder
        self.lstm = nn.LSTM(128 * 56 * 56, hidden_dim, num_layers, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if len(x.size()) == 4:  # 4차원 입력을 받았을 경우
            x = x.unsqueeze(1)  # (batch_size, 1, C, H, W)로 변환
        batch_size, seq_len, C, H, W = x.size()

        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(c_out)

        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out
