# ultra_simple.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

print("=== ENTRAÎNEMENT ET TEST ULTRA-SIMPLE ===")

# 1. Charger les données
print("1. Chargement des données...")
train_data = torch.load('datasets/processed/train_data.pt')
val_data = torch.load('datasets/processed/val_data.pt')
test_data = torch.load('datasets/processed/test_data.pt')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

print(f"   Train: {X_train.shape}")
print(f"   Validation: {X_val.shape}")
print(f"   Test: {X_test.shape}")

# 2. Créer un modèle simple
print("\n2. Création du modèle...")
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))

model = SimpleModel(X_train.shape[1])
print(f"   Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")

# 3. Entraînement
print("\n3. Entraînement...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
batch_size = 32
n_samples = len(X_train)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Mini-batch
    indices = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X_train[batch_idx]
        batch_y = y_train[batch_idx]
        
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_preds = (val_outputs >= 0.5).float()
        val_acc = accuracy_score(y_val.numpy(), val_preds.numpy())
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {total_loss/(n_samples/batch_size):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 4. Test
print("\n4. Évaluation sur test set...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_preds = (test_outputs >= 0.5).float()
    
    test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())
    cm = confusion_matrix(y_test.numpy(), test_preds.numpy())
    
print(f"\n   Résultats sur Test Set:")
print(f"   Accuracy: {test_acc:.4f}")
print(f"\n   Matrice de confusion:")
print(f"   {cm}")
print(f"\n   Rapport de classification:")
print(classification_report(y_test.numpy(), test_preds.numpy(), target_names=['Sporadic', 'Familial']))

# 5. Sauvegarder
print("\n5. Sauvegarde du modèle...")
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': X_train.shape[1],
    'config': {
        'hidden_layers': [64, 32],
        'dropout_rate': 0.3,
        'activation': 'relu'
    }
}, 'models/simple_model.pth')

print(f"✅ Modèle sauvegardé: models/simple_model.pth")

# 6. Charger et vérifier
print("\n6. Vérification du modèle sauvegardé...")
checkpoint = torch.load('models/simple_model.pth', map_location='cpu')
print(f"   Contenu: {list(checkpoint.keys())}")
print(f"   ✓ Modèle prêt pour le rapport")