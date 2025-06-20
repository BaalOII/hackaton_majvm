import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.3):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.out(x))

class TorchModelRunner:
    def __init__(self, X_train, X_test, y_train, y_test, plot_dir="plots",
                 epochs=50, lr=0.001, batch_size=32, verbose=False, patience=7,
                 dropout=0.3, val_split=0.2):
        self.X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        self.X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        self.y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

        self.plot_dir = plot_dir
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.val_split = val_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.X_train.shape[1]
        self.model = PyTorchMLP(input_dim, dropout=dropout).to(self.device)

    def run(self):
        os.makedirs(self.plot_dir, exist_ok=True)

        # Prepare datasets and loaders
        X_tensor = torch.tensor(self.X_train.astype(np.float32))
        y_tensor = torch.tensor(self.y_train.astype(np.float32)).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)

        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / train_size
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_running_loss += loss.item() * xb.size(0)
            val_loss = val_running_loss / val_size
            val_losses.append(val_loss)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save checkpoint
                torch.save(self.model.state_dict(), os.path.join(self.plot_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model before evaluation
        self.model.load_state_dict(torch.load(os.path.join(self.plot_dir, "best_model.pth")))
        self.model.eval()

        # Evaluate on test set
        X_test_tensor = torch.tensor(self.X_test.astype(np.float32)).to(self.device)
        with torch.no_grad():
            probs = self.model(X_test_tensor).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

        acc = np.mean(preds == self.y_test)
        roc = roc_auc_score(self.y_test, probs)
        f1 = classification_report(self.y_test, preds, output_dict=True)['weighted avg']['f1-score']

        model_name = "TorchMLP"

        # Plot training curves
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves: {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'training_curves_{model_name}.png'))
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, preds)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend()
        plt.title(f'ROC Curve: {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'roc_curve_{model_name}.png'))
        plt.close()

        return {
            model_name: {
                "Accuracy Mean": acc,
                "ROC AUC Mean": roc,
                "F1 Mean": f1
            }
        }
