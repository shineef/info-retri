import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class NCFDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NCFModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_size):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * embedding_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, user_input, book_input):
        user_embedded = self.user_embedding(user_input)
        book_embedded = self.book_embedding(book_input)
        concatenated = torch.cat([user_embedded, book_embedded], dim=-1)
        out = self.fc_layers(concatenated)
        return out.squeeze()

def train_ncf_model(ratings, num_users, num_books, epochs=20, embedding_size=6):

    lr=0.0001
    batch_size=64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')

    #  NaN -> 
    ratings['Book-Rating'].fillna(ratings['Book-Rating'].median(), inplace=True)
    X = ratings[['User-ID', 'ISBN']].values
    y = ratings['Book-Rating'].values

    y = np.nan_to_num(y).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = NCFDataset(X_train, y_train)
    val_dataset = NCFDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = NCFModel(num_users, num_books, embedding_size).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Wrap your loader with tqdm
        for inputs, labels in tqdm(train_loader):
            user_input, book_input = inputs[:, 0].to(device), inputs[:, 1].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(user_input, book_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * user_input.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        # Wrap your loader with tqdm
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                user_input, book_input = inputs[:, 0].to(device), inputs[:, 1].to(device)
                labels = labels.to(device)
                outputs = model(user_input, book_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * user_input.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f'NCF_lr{lr}_bs_{batch_size}.pth')

    # Plot training & validation loss values
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'NCF_lr{lr}_bs_{batch_size}_loss.png')
    plt.show()

    return model

