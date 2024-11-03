import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTMModel(nn.Module):
    """
    sLSTM model used during experimentation 
    """
    def __init__(self, input_size, hidden_size, num_blocks, output_size, context_length, num_heads = 4, dropout = 0.1, proj_factor = 1.3, conv1d=4):
        super(xLSTMModel, self).__init__()
        cfg = xLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    embedding_dim=hidden_size,
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d,
                    dropout=dropout,
                    backend="vanilla",
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    embedding_dim=hidden_size,
                    proj_factor=proj_factor, 
                    act_fn="gelu"
                ),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at="all",
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # batch first
        x = self.input_proj(x)
        out = self.xlstm_stack(x)
        out = self.output_proj(out[:, -1, :]) # use only the last time step for prediction
        return out
    
class LogRes(nn.Module):
    """
    Logistic Regression baseline model. Sigmoid activation is not used here as the loss function applies sigmoid
    """
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x):
        out = self.linear(x[:,-1,:])
        return out
    
class LSTMModel(nn.Module):
    """
    LSTM baseline model
    """
    def __init__(self, input_size, hidden_size, num_blocks, output_size, dropout = 0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
                            input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_blocks, 
                            batch_first=True, 
                            dropout= dropout
                        )
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # batch first
        x = self.input_proj(x)
        out, (hn, cn) = self.lstm(x)
        out = self.output_proj(out[:, -1, :]) # use only the last time step for prediction
        return out