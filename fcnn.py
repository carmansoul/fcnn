import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCNN(nn.Module):
    def __init__(self, input_size=40000*2, num_classes=84, learning_rate=0.01, sample_rate=40000):
        super(FCNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = 84
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate
        self.nyquist_bin = input_size // 2
       
        # Bias term: b(f) = -1 / (2pf)^2
        self.bias = nn.Parameter(torch.zeros(self.nyquist_bin, dtype=torch.float32, device=device))
        # Learnable filters in frequency domain
        self.filters = nn.Parameter(torch.ones(self.nyquist_bin, dtype=torch.float32, device=device))

        # Fully connected layer weights: Initialize fundamental & harmonic frequencies to 1, others to 0.1
        self.fc_layer = nn.Linear(self.nyquist_bin, self.num_classes, bias=False)

    def psd(self, x):
        """Compute Power Spectral Density (PSD)."""
        X_f = torch.fft.fft(x.float())  # Compute FFT
        psd = torch.abs(X_f) ** 2  # Compute squared magnitude (Power Spectral Density)
        assert torch.isfinite(psd).all(), "PSD contains NaNs or infs."
        return psd
    
    def convolution(self, pooled_psd):
        """Apply learnable frequency domain filters."""
        return pooled_psd * self.filters

    def forward(self, x):
        X_f = torch.fft.fft(x.float())  # Full FFT
        psd = torch.abs(X_f) ** 2       # Power spectral density
        psd = psd[:, :self.nyquist_bin]  # Keep only positive frequencies < 20kHz

        filtered = psd * self.filters    # Apply learnable frequency filter
        biased = filtered + self.bias    # Add bias term

        logits = self.fc_layer(biased)   # Fully connected output
        return logits
    
    def compute_loss(self, predictions, labels):
        """Compute cross-entropy loss with correct label format."""
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(predictions, labels)
    
    def train_model(self, X_data, Y_data, epochs=10, batch_size=32):
        """Train model with a new train-test split in each epoch."""
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
    
        num_samples = len(X_data)
        indices = torch.arange(num_samples)

        for epoch in range(epochs):
            # Randomly shuffle indices for new train-test split
            shuffled_indices = torch.randperm(num_samples)

            # Split into 80% training and 20% testing
            split_index = int(num_samples * 0.8)
            train_indices = shuffled_indices[:split_index]
            test_indices = shuffled_indices[split_index:]

            # Subset the data
            X_train, Y_train = X_data[train_indices], Y_data[train_indices]
            X_test, Y_test = X_data[test_indices], Y_data[test_indices]

            # Create new DataLoader for each epoch
            train_dataset = TensorDataset(X_train, torch.argmax(Y_train, dim=1))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = self.compute_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def predict(self, x):
        """Generate predictions."""
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.forward(x).argmax(dim=1)
