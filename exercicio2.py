import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Arquitetura do MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hiperparâmetros
input_size = X_train_tensor.shape[1]
model = MLP(input_size)
lr = 0.001
epochs = 200
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Treinamento
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor)

        val_preds = (val_pred > 0.5).float()
        val_acc = (val_preds == y_val_tensor).float().mean()

        train_preds = (y_pred > 0.5).float()
        train_acc = (train_preds == y_train_tensor).float().mean()

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())

    if (epoch+1) % 10 == 0:
        print(f"Época {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}")

# Gráficos de evolução
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Treino')
plt.plot(val_losses, label='Validação')
plt.title('Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Treino')
plt.plot(val_accs, label='Validação')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Avaliação no Teste
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_pred_test_tensor = (y_test_pred > 0.5).float()

y_true_test = y_test_tensor.numpy()
y_pred_test = y_pred_test_tensor.numpy()

# Acurácia
acc = accuracy_score(y_true_test, y_pred_test)
print(f"Acurácia no teste: {acc:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Diabético (0)', 'Diabético (1)'],
            yticklabels=['Não Diabético (0)', 'Diabético (1)'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.show()

# Relatório
print("\nRelatório de Classificação:")
print(classification_report(y_true_test, y_pred_test, digits=4, zero_division=1))
