from .BasePreprocessor import BasePreprocessor
import pandas as pd
from typing import override

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

from src.utils.context import DataProcessingContext

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__() # type: ignore
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(encoding_dim*3, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*3),
            nn.ReLU(),
            nn.Linear(input_dim*3, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x): # type: ignore
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def perform_autoencoder(df: pd.DataFrame, encoding_dim: int, epochs: int, batch_size: int):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df) # type: ignore
    
    data_tensor = torch.FloatTensor(df_scaled) 
    
    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    
    loss = None
    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step() # type: ignore
            
        if (epoch + 1) % 10 == 0 and loss is not None:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')
                 
    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()
        
    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])
    
    return df_encoded

class AutoEncoderPreprocessor(BasePreprocessor):
    """
    Before this preprocessor: DescribingTimeseriesDataLoader
    """

    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        self.encoding_dim = int(parameters.get('encoding_dim', 50))
        self.epochs = int(parameters.get('epochs', 50))
        self.batch_size = int(parameters.get('batch_size', 32))

    @override
    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = dfs[0]
        df_without_id = df.drop('id', axis=1)
        df_encoded = perform_autoencoder(df_without_id, self.encoding_dim, self.epochs, self.batch_size)
        DataProcessingContext.get_instance()["autoencoder_columns"] = df_encoded.columns.to_list()
        df_encoded['id'] = df['id']
        return [df_encoded]
