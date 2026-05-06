# -*- coding: utf-8 -*-
"""
DeepFM Model Implementation.
Combines Factorization Machines (FM) for low-order feature interactions 
and Deep Neural Networks (DNN) for high-order interactions.

Key Optimizations:
1. Xavier Initialization: Ensures stable weights and faster convergence.
2. Refactored DNN: Structured with nn.Sequential for better readability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=8,
                 hidden_dims=[128, 64], num_classes=1, dropout=[0.2, 0.2], 
                 use_cuda=False):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # 1. FM First Order (Linear part)
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(size, 1) for size in feature_sizes])
        
        # 2. FM Second Order (Interaction part)
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_size) for size in feature_sizes])

        # Xavier Initialization for stable training
        for emb in self.fm_first_order_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_second_order_embeddings:
            nn.init.xavier_uniform_(emb.weight)

        # 3. Deep Component (DNN)
        curr_dim = self.field_size * embedding_size
        layers = []
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[i]))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.deep_module = nn.Sequential(*layers)
        
        self.bias = nn.Parameter(torch.randn(1))
        self.to(self.device)

    def forward(self, Xi, Xv):
        # --- FM Part ---
        # First Order
        fm_first_order_res = [emb(Xi[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order_res = torch.cat(fm_first_order_res, dim=1)
        fm_first_order_res = torch.sum(fm_first_order_res * Xv, dim=1, keepdim=True)

        # Second Order: Implementation of the formula: 0.5 * sum((sum(v*x)^2) - sum((v*x)^2))
        embeddings = [emb(Xi[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        embeddings = torch.stack(embeddings, dim=1) 
        embeddings = embeddings * torch.unsqueeze(Xv, 2)

        sum_of_emb = torch.sum(embeddings, dim=1)
        square_of_sum = torch.pow(sum_of_emb, 2)
        sum_of_square = torch.sum(torch.pow(embeddings, 2), dim=1)
        fm_second_order_res = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # --- Deep Part ---
        deep_input = embeddings.view(embeddings.size(0), -1)
        deep_out = self.deep_module(deep_input)

        total_sum = fm_first_order_res + fm_second_order_res + deep_out + self.bias
        return total_sum.squeeze()

    def fit(self, loader_train, loader_val, optimizer, epochs=10):
        """Standard training loop with validation on each epoch."""
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for xi, xv, y in loader_train:
                xi, xv, y = xi.to(self.device), xv.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self(xi, xv)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            acc, auc = self.evaluate(loader_val)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader_train):.6f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

    def evaluate(self, loader):
        """Calculates Accuracy and AUC (Primary metric for CTR)."""
        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xi, xv, y in loader:
                xi, xv = xi.to(self.device), xv.to(self.device)
                logits = self(xi, xv)
                preds = torch.sigmoid(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        acc = ((np.array(all_preds) > 0.5) == np.array(all_labels)).mean()
        return acc, auc