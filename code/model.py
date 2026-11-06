import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.2,
                 output_size=1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 原生PyTorch LSTM层
        # batch_first=True → 适配你的数据格式 (batch, seq_len, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # 时间序列必须用单向LSTM
        )

        # 全连接层：将LSTM输出映射为最终预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        输入 x 形状: (batch_size, seq_len, input_size) → (64, 24, 2)
        """
        lstm_out, _ = self.lstm(x)  # 输出形状: (64, 24, 64)
        last_time_step = lstm_out[:, -1, :]  # 形状: (64, 64)
        output = self.fc(last_time_step)  # 形状: (64, 1)
        # output = torch.sigmoid(output)
        return output


# ===================== 初始化模型 + 测试 =====================
if __name__ == "__main__":
    # 1. 创建模型
    model = LSTM(input_size=2,  # 固定：你的2个物理特征
                 hidden_size=64,  # 隐藏层维度
                 num_layers=2,  # 2层LSTM
                 dropout=0.2,  # 防止过拟合
                 output_size=1)
    model = model.to("cuda")
    test_input = torch.randn(64, 12, 2).to("cuda")

    pred = model(test_input)

    print(f"模型输入形状: {test_input.shape}")  # (64, 24, 2)
    print(f"模型输出形状: {pred.shape}")  # (64, 1)
    print(pred)