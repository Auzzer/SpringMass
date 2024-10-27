# source SpringMass/bin/activate
import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
num_layers = 1
num_epochs = 100
learning_rate = 0.001

# Generate synthetic data (e.g., a sine wave)
seq_length = 20
num_samples = 1000

x_data = torch.linspace(0, 100, num_samples).to(device)
y_data = torch.sin(x_data).to(device)

# Prepare data for training
def create_inout_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length:i + seq_length + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

data = create_inout_sequences(y_data, seq_length)
train_sequences = [seq[0].unsqueeze(-1) for seq in data]
train_labels = [seq[1] for seq in data]

train_sequences = torch.stack(train_sequences).to(device)
train_labels = torch.stack(train_labels).to(device)

# Model, loss function, and optimizer
model = AutoregressiveLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(train_sequences)
    loss = criterion(outputs, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing (predict the next 50 time steps based on the last part of the training sequence)
model.eval()
predicted = train_sequences[-1].unsqueeze(0).to(device)  # start with the last sequence in training
predictions = []

for _ in range(50):  # predict 50 future time steps
    with torch.no_grad():
        predicted_output = model(predicted)
        predictions.append(predicted_output.item())
        predicted = torch.cat((predicted[:, 1:, :], predicted_output.unsqueeze(1)), dim=1)

print(predictions)
