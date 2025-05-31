import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
import time

# Device setup (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Define spike encoding function
def rate_based_encoding(images, num_steps=10):
    spike_probs = images * 0.8  # Scale pixel values to spike probabilities
    spike_trains = torch.rand_like(spike_probs) < spike_probs
    return spike_trains.float().to(device)

# Define SNN model
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=snn.surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=snn.surrogate.fast_sigmoid())
        self.spike_count = 0  # To count spikes

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 28*28)
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(1)):
            cur_x = x[:, step, :]
            spk1, mem1 = self.lif1(self.fc1(cur_x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            self.spike_count += spk2.sum().item()  # Count spikes

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Custom loss function
def spike_count_loss(output_spikes, labels):
    spike_counts = output_spikes.sum(dim=0)
    target_spikes = torch.zeros_like(spike_counts)
    target_spikes[range(labels.size(0)), labels] = 1
    loss = torch.mean((spike_counts - target_spikes) ** 2)
    return loss

# Initialize model
snn_model = SNN().to(device)

# Loss and optimizer
optimizer = optim.Adam(snn_model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
start_time = time.time()

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        spike_trains = rate_based_encoding(images, num_steps=10)
        optimizer.zero_grad()
        spk_rec, _ = snn_model(spike_trains)
        loss = spike_count_loss(spk_rec, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(trainloader)}")

# Training time
end_time = time.time()
training_time_snn = end_time - start_time

# Evaluate SNN and calculate energy
correct, total = 0, 0
snn_model.spike_count = 0  # Reset spike count

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        spike_trains = rate_based_encoding(images, num_steps=10)
        spk_rec, _ = snn_model(spike_trains)
        _, predicted = torch.max(spk_rec.sum(0), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_snn = 100 * correct / total
energy_per_spike = 1e-12  # Energy per spike in Joules
total_energy = snn_model.spike_count * energy_per_spike  # Total energy for test set

# Print results
print(f"\nSNN Training Time: {training_time_snn:.2f} seconds")
print(f"SNN Test Accuracy: {accuracy_snn:.2f}%")
print(f"Total Spikes Generated: {snn_model.spike_count}")
print(f"Energy per spike: {energy_per_spike} Joules")
print(f"Total Energy Consumption: {total_energy} Joules")