import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torch.functional import F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GenderModelPL(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(128, 128, 5, stride=5 // 2),  # 128 x 128 -> 128 x 62
            nn.ReLU(),
            nn.MaxPool1d(4, 4),  # 128 x 15
            nn.Conv1d(128, 64, 5, stride=5 // 2),  # 64 x 6
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Global Max Pool -> 64 x 1
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        outputs = self(x)
        loss = nn.BCEWithLogitsLoss()(outputs, y)
        preds = torch.round(outputs)
        acc = (preds == y).float().mean()
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'acc': acc, 'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        outputs = self(x)
        loss = nn.MSELoss()(outputs, y)
        preds = torch.round(outputs)
        acc = (preds == y).float().mean()
        tensorboard_logs = {'val_loss': loss, 'val_acc': acc}
        return {'val_acc': acc, 'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.num_layers=3
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, batch_size=1, input_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=512):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = 3
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size + self.output_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.output_size, self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(input)
        combined_inputs = torch.cat((embedded[0], hidden[0]), 1)
        attn_input = self.attn(combined_inputs)
        attn_weights = F.softmax(attn_input, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size=1, input_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)