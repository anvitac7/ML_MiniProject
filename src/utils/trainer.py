import torch
import torch.nn as nn
import numpy as np
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for x, y in self.train_loader:
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'saved_models/best_model.pth')
                print(f"Saved best model with val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break