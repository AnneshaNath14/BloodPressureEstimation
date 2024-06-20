from model import UNet
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data import get_datasets

def train_model(
        dataset_path, 
        device = 'cuda',
        batch_size = 32,
        num_epochs = 10,
        learning_rate = 0.0001
    ):
    train_dataset, test_dataset = get_datasets(dataset_path)
    # Instantiate the model
    model = UNet().to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            ecg = batch['ecg'].unsqueeze(1).to(device)  # Add channel dimension
            ppg = batch['ppg'].unsqueeze(1).to(device)  # Add channel dimension

            # Ensure that tensors have at least three dimensions
            if ecg.dim() < 3 or ppg.dim() < 3:
                raise ValueError("ECG and PPG tensors must have at least three dimensions.")

            # Pad the shorter tensor to match the length of the longer tensor
            max_length = max(ecg.shape[2], ppg.shape[2])
            ecg = torch.nn.functional.pad(ecg, (0, max_length - ecg.shape[2]))
            ppg = torch.nn.functional.pad(ppg, (0, max_length - ppg.shape[2]))

            inputs = torch.cat((ecg, ppg), dim=1)  # Concatenate along the channel dimension
            targets = batch['abp']

            #print(f'Shapes - Input: {inputs.shape} Targets: {targets.shape} Outputs: {model(inputs).shape}')

            optimizer.zero_grad()
            # Inside your training loop
            outputs = model(inputs)

            # Squeeze the target to match the output size
            targets_squeezed = torch.squeeze(targets, dim=1).to(device)

            # Add a new dimension to targets_squeezed
            targets_squeezed = targets_squeezed.unsqueeze(1)

            # If your criterion is mean squared error (MSE) loss
            loss = criterion(outputs, targets_squeezed)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_model(
        dataset_path = "dataset/", 
        device = 'cpu',
        batch_size = 32,
        num_epochs = 10,
        learning_rate = 0.0001
    )