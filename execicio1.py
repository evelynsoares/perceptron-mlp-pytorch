# Preparação dos Dados para MLP com PyTorch - Diagnóstico de Diabetes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# --- Carregamento do Dataset ---
url = "https://raw.githubusercontent.com/pcbrom/perceptron-mlp-cnn/refs/heads/main/data/diabetes.csv"
df = pd.read_csv(url)

# --- Tratamento de Dados Ausentes ---
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].fillna(df[cols_with_invalid_zeros].median())

# --- Visualizações ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Histogramas
_ = df.hist(figsize=(14, 10), bins=20, edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots por Outcome
plt.figure(figsize=(18, 6))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BMI', 'Age']):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'{col} por Outcome')
plt.tight_layout()
plt.show()

# --- Divisão dos Dados ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# --- Padronização ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Conversão para Tensores PyTorch ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Shapes finais (para checagem - podem ser comentados no relatório final)
print("Shape X_train:", X_train_tensor.shape)
print("Shape y_train:", y_train_tensor.shape)
print("Shape X_val:", X_val_tensor.shape)
print("Shape y_val:", y_val_tensor.shape)
print("Shape X_test:", X_test_tensor.shape)
print("Shape y_test:", y_test_tensor.shape)
