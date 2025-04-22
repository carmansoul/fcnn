import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from fcnn import FCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, sample_rate=40000, duration=2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.length = int(sample_rate * duration)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freqs = torch.tensor([440.0 * 2 ** ((k - 46) / 12) for k in range(84)],
                                  dtype=torch.float32, device=self.device)

    def normalize_energy(self, x, target_energy):
        energy = torch.sum(x ** 2)
        if energy < 1e-10:
            return x
        return x * torch.sqrt(torch.tensor(target_energy, device=x.device) / energy)

    def bandlimited_wiener(self, f_cutoff, length):
        noise = torch.randn(length, device=self.device)
        spectrum = torch.fft.fft(noise)
        freqs = torch.fft.fftfreq(length, d=1 / self.sample_rate).to(self.device)
        mask = torch.abs(freqs) <= f_cutoff
        spectrum[~mask] = 0
        env = torch.fft.irfft(spectrum, n=length)
        return env

    def __getitem__(self, idx):
        pitch_idx = idx % 84
        f0 = self.freqs[pitch_idx]
        label = pitch_idx  # return as integer class

        t = torch.linspace(0, self.duration, self.length, device=self.device)
        waveform = torch.zeros_like(t)

        for n in range(1, 6):
            f = f0 * n
            if f > self.sample_rate / 2:
                continue
            A_t = self.bandlimited_wiener(f * 0.99, self.length)
            A_t = self.normalize_energy(A_t, 1.0)
            phase = torch.rand(1, device=self.device) * 2 * torch.pi
            waveform += A_t * torch.sin(2 * torch.pi * f * t + phase)

        waveform = self.normalize_energy(waveform, 1.0)

        # Add noise with SNR = 10:1
        snr_linear = 10
        noise_energy = 1 / snr_linear
        noise = torch.cumsum(torch.randn(self.length, device=self.device), dim=0)
        noise = self.normalize_energy(noise, noise_energy)
        waveform += noise

        # Don't normalize again here; we want the SNR preserved
        return waveform.float(), label

    def __len__(self):
        return 21504

# Define note mapping
OCTAVES = list(map(str, range(1, 8)))
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_labels = [f"{p}{o}" for o in range(1, 8) for p in PITCHES]
NOTE_TO_INDEX = {f"{p}{o}": i for i, (p, o) in enumerate([(p, o) for o in OCTAVES for p in PITCHES])}

# Replace X_data / Y_data with procedural synth dataset
synth_dataset = SynthDataset(duration=2.0, sample_rate=40000)
train_size = int(0.8 * len(synth_dataset))
test_size = len(synth_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(synth_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=21, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=21)

# Model parameters
target_length = 80000
input_size = target_length // 2
num_classes = 84
learning_rate = 0.01

fcnn = FCNN(input_size, num_classes=num_classes, learning_rate=learning_rate).to(device)
optimizer = torch.optim.SGD(fcnn.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
early_stop_patience = 3
zero_loss_epochs = 0

for epoch in range(num_epochs):
    if epoch == 49:
        fcnn.eval()
        true_labels_50, predicted_labels_50 = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = fcnn.predict(batch_x)
                predicted_notes = y_pred.cpu().numpy()
                true_notes = batch_y.cpu().numpy()
                true_labels_50.extend(true_notes)
                predicted_labels_50.extend(predicted_notes)

        conf_matrix_50 = confusion_matrix(true_labels_50, predicted_labels_50)
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_50, annot=False, fmt="d", cmap="Blues",
                    xticklabels=note_labels, yticklabels=note_labels)
        plt.xlabel("Predicted Pitch + Octave")
        plt.ylabel("True Pitch + Octave")
        plt.title("Confusion Matrix After Epoch 50")
        plt.savefig("/home/cgreenway/FCNN/heatmap_epoch50.png")
        plt.close()

    torch.manual_seed(41 + epoch)
    np.random.seed(41 + epoch)
    fcnn.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        y_pred = fcnn(batch_x)
        loss = fcnn.compute_loss(y_pred, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(fcnn.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    if avg_loss < 1e-6:
        zero_loss_epochs += 1
        if zero_loss_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {zero_loss_epochs} zero-loss epochs.")
            break
    else:
        zero_loss_epochs = 0

# Final Evaluation
fcnn.eval()
true_labels, predicted_labels = [], []
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_pred = fcnn.predict(batch_x)
        predicted_notes = y_pred.cpu().numpy()
        true_notes = batch_y.cpu().numpy()
        true_labels.extend(true_notes)
        predicted_labels.extend(predicted_notes)
        correct += (predicted_notes == true_notes).astype(int).sum()
        total += batch_y.size(0)
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

conf_matrix_np = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_np, annot=False, fmt="d", cmap="Blues",
            xticklabels=note_labels, yticklabels=note_labels)
plt.xlabel("Predicted Pitch + Octave")
plt.ylabel("True Pitch + Octave")
plt.title("Pitch + Octave Classification Heatmap")
HEATMAP_FILE = "/home/cgreenway/FCNN/heatmap.png"
plt.savefig(HEATMAP_FILE)
plt.show()
print(f"Heatmap saved at {HEATMAP_FILE}")
