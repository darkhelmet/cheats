# PyTorch

## Installation
```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Intel GPU (XPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Development version (nightly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Import Essentials
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
```

## Tensor Basics

### Creating Tensors
```python
# From data
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Zeros and ones
x = torch.zeros(3, 4)
x = torch.ones(2, 3)
x = torch.eye(3)  # identity matrix

# Random tensors
x = torch.randn(2, 3)  # normal distribution
x = torch.rand(2, 3)   # uniform [0, 1)
x = torch.randint(0, 10, (2, 3))  # random integers

# From numpy
numpy_array = np.array([1, 2, 3])
x = torch.from_numpy(numpy_array)

# Ranges
x = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1.0]
```

### Tensor Properties
```python
x = torch.randn(3, 4, 5)

print(x.shape)      # torch.Size([3, 4, 5])
print(x.size())     # torch.Size([3, 4, 5])
print(x.dtype)      # torch.float32
print(x.device)     # cpu or cuda:0
print(x.ndim)       # 3
print(x.numel())    # 60 (total elements)
```

### Tensor Operations
```python
# Arithmetic
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z = x + y           # or torch.add(x, y)
z = x - y           # or torch.sub(x, y)
z = x * y           # element-wise multiplication
z = x / y           # element-wise division
z = x @ y           # dot product
z = torch.matmul(x, y)  # matrix multiplication

# In-place operations (end with _)
x.add_(1)           # adds 1 to x in-place
x.mul_(2)           # multiplies x by 2 in-place
```

### Reshaping and Indexing
```python
x = torch.randn(4, 4)

# Reshaping
x = x.view(16)      # or x.view(-1)
x = x.view(2, 8)
x = x.reshape(4, 4) # more flexible than view
x = x.squeeze()     # remove dimensions of size 1
x = x.unsqueeze(0)  # add dimension at index 0

# Indexing
x[0, 1]             # element at row 0, col 1
x[:, 1]             # all rows, column 1
x[1, :]             # row 1, all columns
x[0:2, 1:3]         # submatrix

# Advanced indexing
mask = x > 0
x[mask]             # elements where mask is True
```

### Device Management
```python
# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move tensors to device
x = torch.randn(3, 3)
x = x.to(device)
# or
x = x.cuda() if torch.cuda.is_available() else x

# Create tensors directly on device
x = torch.randn(3, 3, device=device)
```

## Neural Networks

### Basic Neural Network
```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet(784, 128, 10)
print(model)
```

### Convolutional Neural Network
```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Common Layers
```python
# Linear layers
nn.Linear(in_features, out_features)
nn.Linear(784, 10, bias=False)

# Convolutional layers
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.Conv2d(1, 32, 3, stride=1, padding=1)
nn.Conv3d(1, 16, 3)

# Pooling layers
nn.MaxPool2d(kernel_size=2)
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))

# Normalization
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Activation functions
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)

# Regularization
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)
```

## Loss Functions
```python
# Classification
criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()  # Binary cross entropy
criterion = nn.BCEWithLogitsLoss()  # BCE with sigmoid

# Regression
criterion = nn.MSELoss()  # Mean squared error
criterion = nn.L1Loss()   # Mean absolute error
criterion = nn.SmoothL1Loss()  # Huber loss

# Custom loss example
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)
```

## Optimizers
```python
# Common optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

## Training Loop
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Optional: step scheduler
        # scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {epoch_loss:.4f}')

# Usage
train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)
```

## Evaluation
```python
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / total
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    
    return accuracy, avg_loss
```

## Data Loading

### Custom Dataset
```python
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Usage
dataset = CustomDataset(data, labels, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Data Transforms
```python
from torchvision import transforms

# Common transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# For evaluation (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## Model Saving and Loading
```python
# Save model
torch.save(model.state_dict(), 'model_weights.pth')
torch.save(model, 'complete_model.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load model
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model = torch.load('complete_model.pth', map_location=device)

# Load checkpoint
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Autograd and Gradients
```python
# Enable/disable gradients
x = torch.randn(3, requires_grad=True)

# Forward pass
y = x.sum()

# Backward pass
y.backward()
print(x.grad)

# Gradient context managers
with torch.no_grad():
    # Operations here won't track gradients
    y = model(x)

# Temporarily enable gradients
with torch.enable_grad():
    # Operations here will track gradients
    pass

# Manual gradient computation
def custom_backward(x):
    x.retain_grad()
    y = x ** 2
    y.backward(torch.ones_like(y))
    return x.grad
```

## Mixed Precision Training
```python
from torch.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

def train_with_amp(model, train_loader, criterion, optimizer, device):
    model.train()
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Use autocast for forward pass
        with autocast(device_type='cuda'):
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## torch.compile (PyTorch 2.0+)
```python
# Optimize model with torch.compile
model = torch.compile(model)

# With specific backend
model = torch.compile(model, backend="inductor")

# For inference only
@torch.compile
def inference_function(x):
    return torch.sin(x).cos()

# Disable compilation for debugging
model = torch.compile(model, disable=True)
```

## Model Utilities

### Parameter Counting
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters")
```

### Model Summary
```python
def model_summary(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params": sum([param.nelement() for param in module.parameters()])
            }
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    device = next(model.parameters()).device
    summary = {}
    hooks = []
    
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.randn(*input_size).to(device)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return summary
```

## Transfer Learning
```python
import torchvision.models as models

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Only train final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tuning: unfreeze some layers
for param in model.layer4.parameters():
    param.requires_grad = True
```

## Common Patterns

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
```

### Gradient Clipping
```python
# During training
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Learning Rate Finding
```python
def find_lr(model, train_loader, optimizer, criterion, device):
    lrs = []
    losses = []
    lr = 1e-7
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        lr *= 1.1
        if lr > 1:
            break
    
    return lrs, losses
```

## Debugging Tips

### Check for NaN/Inf
```python
def check_for_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
```

### Monitor Gradients
```python
def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: {grad_norm:.4f}")
```

### Memory Usage
```python
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

## Performance Tips

- Use `torch.compile()` for PyTorch 2.0+ performance gains
- Use mixed precision training with AMP for faster training
- Set `torch.backends.cudnn.benchmark = True` for consistent input sizes
- Use `pin_memory=True` in DataLoader for faster GPU transfer
- Use appropriate `num_workers` in DataLoader (typically 2-4x number of GPUs)
- Use `torch.no_grad()` during inference to save memory
- Consider using `torch.jit.script()` for model optimization
- Use `torch.utils.checkpoint` for memory-efficient training of large models