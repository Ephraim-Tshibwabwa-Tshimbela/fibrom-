import torch
import torch.nn as nn
import torch.optim as optim

# Charger les données
train_data = torch.load('datasets/processed/train_data.pt')
val_data = torch.load('datasets/processed/val_data.pt')

print(f"Données chargées:")
print(f"  Train: {train_data['X'].shape}")
print(f"  Validation: {val_data['X'].shape}")

# Créer un modèle très simple pour tester
class TestModel(nn.Module):
    def __init__(self, input_size):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# Tester la dimension
input_size = train_data['X'].shape[1]
print(f"\nCréation du modèle avec input_size={input_size}")

model = TestModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Test d'un seul batch
batch_size = 32
X_batch = train_data['X'][:batch_size]
y_batch = train_data['y'][:batch_size]

print(f"\nTest avec batch de {batch_size} échantillons:")
print(f"  X_batch shape: {X_batch.shape}")
print(f"  y_batch shape: {y_batch.shape}")

# Forward pass
outputs = model(X_batch)
loss = criterion(outputs, y_batch)

print(f"  Outputs shape: {outputs.shape}")
print(f"  Loss: {loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("\n✅ Test réussi! Les dimensions sont correctes.")
print("\nVous pouvez maintenant exécuter l'entraînement complet.")