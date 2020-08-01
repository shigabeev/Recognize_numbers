import torch
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import GenderDataLoader, generate_embedding
from models import GenderModelPL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    df = pd.read_csv('train_norm.csv')
    train_X, validation_X, train_y, validation_y = train_test_split(df.path, df.gender)

    params = {'batch_size': 64,
                  'num_workers': 6}

    training_set = GenderDataLoader(train_X, train_y)
    train_loader = DataLoader(training_set, **params, shuffle=True)

    validation_set = GenderDataLoader(validation_X, validation_y)
    validation_loader = DataLoader(training_set,  **params, shuffle=False)

    gender_model = GenderModelPL()
    trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=10)
    trainer.fit(gender_model, train_loader, val_dataloaders=validation_loader)

    torch.save(gender_model.state_dict(), 'states/gender_model_.weights')