import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import torch
except ImportError:
    print("Installing torch...")
    install("torch")
    install("torchvision")
    install("torchaudio")

try:
    import numpy
except ImportError:
    print("Installing numpy...")
    install("numpy")

try:
    import matplotlib
except ImportError:
    print("Installing matplotlib...")
    install("matplotlib")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

timesteps = 500
t = np.arange(timesteps)

temperature = 20 + 2*np.sin(0.02*t) + np.random.normal(0, 0.2, timesteps)
battery = 3.7 + 0.01*np.sin(0.01*t) + np.random.normal(0, 0.01, timesteps)
gyro = np.sin(0.05*t) + np.random.normal(0, 0.05, timesteps)
power = 50 + 5*np.sin(0.015*t) + np.random.normal(0, 0.5, timesteps)

temperature[200:210] += 10
battery[300:350] -= np.linspace(0, 1, 50)
gyro[400:420] = 0
power[150:155] += 20

data = np.stack([temperature, battery, gyro, power], axis=1)
sensor_names = ["Temperature", "Battery", "Gyroscope", "Power"]

mean = data.mean(axis=0)
std = data.std(axis=0)
data_norm = (data - mean) / std

window_size = 20

def create_windows(data, window):
    return np.array([data[i:i+window] for i in range(len(data)-window)])

X = create_windows(data_norm, window_size)
print("Input shape:", X.shape)

X_tensor = torch.tensor(X, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        attn_output, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )
        out = self.ff(attn_output)
        return out, attn_weights

model = TransformerModel(input_dim=4)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output, _ = model(X_tensor)
    loss = loss_fn(output, X_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    recon, attn_weights = model(X_tensor)

errors = ((recon - X_tensor)**2).mean(dim=(1,2)).numpy()
threshold = errors.mean() + 2*errors.std()
anomalies = np.where(errors > threshold)[0]

def get_sensor_importance(attn, input_seq, recon_seq, index):
    attn_sample = attn[index].mean(dim=0).detach().numpy()
    time_importance = attn_sample.mean(axis=0)
    recon_error = ((recon_seq[index] - input_seq[index])**2).mean(dim=0).detach().numpy()
    importance = recon_error * time_importance.mean()
    return importance

plt.figure()
for i in range(4):
    plt.plot(data[:, i], label=sensor_names[i])
plt.legend()
plt.title("Telemetry Signals")
plt.show()

plt.figure()
plt.plot(errors, label="Anomaly Score")
plt.axhline(threshold, linestyle='--', label="Threshold")
plt.scatter(anomalies, errors[anomalies])
plt.legend()
plt.title("Anomaly Detection")
plt.show()

if len(anomalies) > 0:
    idx = anomalies[0]
    attn_map = attn_weights[idx].mean(dim=0).detach().numpy()

    plt.figure()
    plt.imshow(attn_map)
    plt.colorbar()
    plt.title("Attention Heatmap")
    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.show()

for idx in anomalies[:5]:
    importance = get_sensor_importance(attn_weights, X_tensor, recon, idx)
    top_sensor = sensor_names[np.argmax(importance)]

    print(f"Anomaly detected at timestep {idx}")
    print(f"Most contributing sensor: {top_sensor}")
    print("-----")

def explain_anomaly(index):
    if index >= len(errors):
        print("Invalid index")
        return

    print(f"\n--- Explanation for anomaly at index {index} ---")
    print(f"Anomaly score: {errors[index]:.4f}")

    importance = get_sensor_importance(attn_weights, X_tensor, recon, index)
    sorted_idx = np.argsort(-importance)

    for i in sorted_idx:
        print(f"{sensor_names[i]} contribution: {importance[i]:.4f}")

if len(anomalies) > 0:
    explain_anomaly(anomalies[0])
