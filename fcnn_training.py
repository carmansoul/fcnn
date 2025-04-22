import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import soundfile as sf
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from fcnn import FCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define note mapping
OCTAVES = list(map(str, range(1, 8)))
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_INDEX = {f"{p}{o}": i for i, (p, o) in enumerate([(p, o) for o in OCTAVES for p in PITCHES])}

# Audio processing functions
def extract_note_name(filename):
    return filename.split("_", 1)[0]

def filename_to_one_hot(note_name):
    """Convert note name to one-hot encoded tensor"""
    if note_name not in NOTE_TO_INDEX:
        raise ValueError(f"Unexpected note name: {note_name}")
    note_index = NOTE_TO_INDEX[note_name]
    one_hot = torch.zeros(84, dtype=torch.float32)
    one_hot[note_index] = 1
    return one_hot

def set_dataset(X, Y, test_size=0.2):
    """Split dataset into training and test sets"""
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], Y[:split_index], X[split_index:], Y[split_index:]

# Paths
AUDIO_DIR = "/home/cgreenway/FCNN/Chunked_Samples"

def load_audio_files(directory, target_length=80000, sr=40000):
    """Load and preprocess audio files into consistent-length waveforms"""
    audio_data, labels, filenames = [], [], []
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            try:
                y, actual_sr = sf.read(file_path, dtype='float32')
                y = torch.tensor(y, dtype=torch.float32)

                # Resample if needed
                if actual_sr != sr:
                    raise ValueError(f"Sample rate mismatch: expected {sr}, got {actual_sr}")

                # Pad or crop to match target length
                if len(y) < target_length:
                    y = torch.nn.functional.pad(y, (0, target_length - len(y)))
                else:
                    y = y[:target_length]

                # Label extraction
                note_name = extract_note_name(file)
                one_hot_label = filename_to_one_hot(note_name)

                audio_data.append(y)
                labels.append(one_hot_label)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return torch.stack(audio_data), torch.stack(labels), filenames

# Load dataset
X_data, Y_data, filenames = load_audio_files(AUDIO_DIR, target_length=80000, sr=40000)
if len(X_data) == 0 or len(Y_data) == 0:
    raise RuntimeError("No data loaded. Check your AUDIO_DIR and file formats.")

# Split into train/test
X_train, Y_train, X_test, Y_test = set_dataset(X_data, Y_data, test_size=0.2)

# Convert one-hot labels to class indices
train_dataset = TensorDataset(X_train, torch.argmax(Y_train, dim=1))
test_dataset = TensorDataset(X_test, torch.argmax(Y_test, dim=1))

# Define DataLoaders (AFTER datasets are defined)
batch_size = 21
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model parameters
target_length = 80000  # 2 seconds of audio
input_size = target_length // 2
num_classes = 84
learning_rate = 0.01

# Initialize FCNN model
fcnn = FCNN(input_size, num_classes=num_classes, learning_rate=learning_rate).to(device)
optimizer = torch.optim.SGD(fcnn.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
# Early stopping config
early_stop_patience = 3
zero_loss_epochs = 0
for epoch in range(num_epochs):
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

    # Early stopping logic
    if avg_loss < 1e-6:  # Essentially zero
        zero_loss_epochs += 1
        if zero_loss_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {zero_loss_epochs} zero-loss epochs.")
            break
    else:
        zero_loss_epochs = 0  # Reset if loss is non-zero

# Evaluation
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

        correct += (predicted_notes == true_notes).sum()
        total += batch_y.size(0)  # Correct total count

        print(f"Sample prediction: {predicted_notes[0]}, True label: {true_notes[0]}")
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Compute confusion matrix
conf_matrix_np = confusion_matrix(true_labels, predicted_labels)

# Define tick labels for heatmap (pitch names with octaves)
note_labels = [f"{p}{o}" for o in range(1, 8) for p in PITCHES]  # Ensures correct order

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_np, annot=False, fmt="d", cmap="Blues",
            xticklabels=note_labels, yticklabels=note_labels)

plt.xlabel("Predicted Pitch + Octave")
plt.ylabel("True Pitch + Octave")
plt.title("Pitch + Octave Classification Heatmap")

# Save the heatmap
HEATMAP_FILE = "/home/cgreenway/FCNN/heatmap.png"
plt.savefig(HEATMAP_FILE)
plt.show()

print(f"Heatmap saved at {HEATMAP_FILE}")
