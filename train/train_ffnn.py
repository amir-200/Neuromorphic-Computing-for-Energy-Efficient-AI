import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Define transformation for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)     # Second layer
        self.fc3 = nn.Linear(64, 10)      # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to calculate FLOPs
def calculate_flops(model, input_size):
    flops = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            flops += 2 * layer.in_features * layer.out_features
    return flops

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Calculate FLOPs and energy per image
input_size = 28 * 28
flops_per_image = calculate_flops(model, input_size)
energy_per_flop = 1e-9  # Energy per FLOP in Joules
energy_per_image = flops_per_image * energy_per_flop

# Training the NN and recording time
num_epochs = 5
start_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")

end_time = time.time()
training_time = end_time - start_time

# Evaluate accuracy and calculate total energy
correct = 0
total = 0
total_energy = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_energy += energy_per_image * images.size(0)  # Energy for this batch

accuracy = 100 * correct / total

# Print results
print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"FLOPs per image: {flops_per_image}")
print(f"Energy per image: {energy_per_image} Joules")
print(f"Total Energy Consumption: {total_energy} Joules")