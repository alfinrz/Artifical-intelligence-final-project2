import torch
import torch.nn as nn
import torch.optim as optim

from metrics import *
from ped_dataset import *

class RunConfig:
    sequence_length = 20
    min_sequence_length = 10
    observed_history = 8

    sample = False
    num_samples = 20
    sample_angle_std = 25

    dataset_paths = [
                     "./data/eth_univ",
                     "./data/eth_hotel",
                     "./data/ucy_zara01",
                     "./data/ucy_zara02",
                     "./data/ucy_univ"
                    ]

def load_datasets():
    datasets = []
    datasets_size = 0
    for dataset_path in RunConfig.dataset_paths:
        if 'HOME' not in os.environ:
            os.environ['HOME'] = os.environ['USERPROFILE']
        dataset_path = dataset_path.replace('~', os.environ['HOME'])
        print("Loading dataset {}".format(dataset_path))
        dataset = PedDataset(dataset_path=dataset_path, sequence_length=RunConfig.sequence_length, observed_history=RunConfig.observed_history, \
                              min_sequence_length=RunConfig.min_sequence_length)
        datasets.append(dataset)
        datasets_size += len(dataset)
    print("Size of all datasets: {}".format(datasets_size))
    return datasets

class ConstantVelocityModel(nn.Module):
    def __init__(self):
        super(ConstantVelocityModel, self).__init__()

    def forward(self, observed):
        obs_rel = observed[:, 1:] - observed[:, :-1]
        deltas = obs_rel[:, -1].unsqueeze(1)
        y_pred_rel = deltas.repeat(1, 12, 1)
        return y_pred_rel

def train_cvm_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            observed = batch_x.permute(1, 0, 2)
            y_true_rel, masks = batch_y
            y_true_rel = y_true_rel.permute(1, 0, 2)

            y_pred_rel = model(observed)
            loss = criterion(y_pred_rel, y_true_rel)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

# Use this function to create a DataLoader for your training dataset
def create_train_loader(trainset, batch_size):
    return torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

def main():
    # Load datasets
    datasets = load_datasets()

    # Choose a dataset for training (modify this part as needed)
    train_dataset = datasets[0]
    train_loader = create_train_loader(train_dataset, batch_size=1)

    # Initialize the CVM model, criterion, and optimizer
    cvm_model = ConstantVelocityModel()
    criterion = nn.MSELoss()  # You may need to choose an appropriate loss function
    optimizer = optim.Adam(cvm_model.parameters(), lr=0.001)  # Adjust the learning rate as needed

    # Train the CVM model
    train_cvm_model(cvm_model, train_loader, criterion, optimizer, num_epochs=10)

    # Now you can use the trained model for evaluation or other tasks