import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from aidia import APP_DIR
from aidia.ai.config import AIConfig

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TestModel():
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = None
        self.stop_flag = False
    
    def set_config(self, config):
        self.config = config

    def build_dataset(self, mode=None):
        with np.load(os.path.join(APP_DIR, 'ai', 'data', 'mnist.npz'), allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
        
        # データを正規化 (0-255 -> 0-1)
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        
        # PyTorchテンソルに変換
        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train).long()
        
        # データセットを作成
        full_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
        # 訓練データと検証データに分割 (80:20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # DataLoaderを作成
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.total_batchsize, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.total_batchsize, 
            shuffle=False
        )
        
        self.dataset = [(x_train, y_train), (x_test, y_test)]

    def build_model(self, mode=None):
        self.model = MNISTModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, custom_callbacks=None):
        self.stop_flag = False
        
        for epoch in range(self.config.EPOCHS):
            if self.stop_flag:
                break
                
            # トレーニングフェーズ
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.stop_flag:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # 検証フェーズ
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # メトリクスの計算
            train_loss = train_loss / len(self.train_loader)
            train_acc = train_correct / train_total
            val_loss = val_loss / len(self.val_loader)
            val_acc = val_correct / val_total
            
            # カスタムコールバックの実行
            if custom_callbacks:
                logs = {
                    'loss': train_loss,
                    'accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                for callback in custom_callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, logs)

    def save(self):
        pass

    def stop_training(self):
        self.stop_flag = True