import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Define the neural network model
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        activation = x.detach().clone()  # Capture activations for visualization
        x = self.fc2(x)
        return x, activation

# Generate random input and labels (modify this based on your actual data)
input_data = torch.randn(10, 4*5*3)  # 10 samples of image data (4x5x3)
labels = torch.randint(0, 3, (10,))  # Random labels [0, 1, 2] for 10 samples

# Define the model, loss function, optimizer
input_size = 4*5*3  # Change this based on your input size
hidden_size = 100
output_size = 3  # Number of output classes
model = TwoLayerNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set up TensorBoard writer
writer = SummaryWriter()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs, activations = model(input_data)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)

    # Write training loss and accuracy to TensorBoard
    writer.add_scalar('Training Loss', loss.item(), epoch)
    writer.add_scalar('Training Accuracy', accuracy, epoch)

    # Visualize activations in TensorBoard
    writer.add_histogram('Hidden Layer Activations', activations, epoch)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pth')

# Close TensorBoard writer
writer.close()




# Modify the forward method in the TwoLayerNet class
class TwoLayerNet(nn.Module):
    # ... (unchanged)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        activation = x.detach().clone()  # Capture activations for visualization
        x = self.fc2(x)
        return x, activation

# Within the training loop after the forward pass
outputs, activations = model(input_data)

# Visualize activations in TensorBoard
for i in range(len(activations)):
    writer.add_histogram(f'Activation_Layer_{i}', activations[i], epoch)
