# TorchVision

TorchVision is PyTorch's computer vision library, providing datasets, model architectures, and common image transformations for computer vision tasks. It includes pre-trained models, data loading utilities, and transforms for efficient computer vision pipelines.

## Installation

```bash
# Basic installation with PyTorch
pip install torch torchvision

# With CUDA support (check PyTorch website for correct version)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Development version
pip install git+https://github.com/pytorch/vision.git

# With additional dependencies
pip install torchvision pillow matplotlib opencv-python
```

## Basic Setup

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Check versions
print(f"TorchVision version: {torchvision.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

## Core Functionality

### Image Transforms

```python
# Basic transforms
transform = transforms.Compose([
    transforms.Resize(256),                    # Resize to 256x256
    transforms.CenterCrop(224),               # Center crop to 224x224
    transforms.ToTensor(),                    # Convert PIL to tensor [0,1]
    transforms.Normalize(                     # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),         # Random crop and resize
    transforms.RandomHorizontalFlip(p=0.5),   # Random horizontal flip
    transforms.ColorJitter(                   # Random color changes
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomRotation(10),            # Random rotation ±10 degrees
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply transforms to an image
image = Image.open("path/to/image.jpg")
transformed_image = transform(image)
print(f"Original size: {image.size}")
print(f"Transformed shape: {transformed_image.shape}")  # [C, H, W]
```

### Advanced Transforms

```python
# Geometric transforms
geometric_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,                           # Rotation
        translate=(0.1, 0.1),                # Translation
        scale=(0.8, 1.2),                    # Scale
        shear=10                             # Shear
    ),
    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.5
    ),
    transforms.ElasticTransform(alpha=250.0, sigma=5.0)  # Elastic deformation
])

# Advanced color transforms
color_transforms = transforms.Compose([
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomEqualize(p=0.5),
    transforms.RandomPosterize(bits=2, p=0.5),
    transforms.RandomSolarize(threshold=128, p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
])

# Cutout/Erasing augmentation
erase_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0
    )
])

# Mix multiple transforms
strong_augment = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    color_transforms,
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Built-in Datasets

```python
# CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

cifar10_test = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform
)

# ImageNet dataset (requires downloaded data)
imagenet_train = datasets.ImageNet(
    root='./data/imagenet',
    split='train',
    transform=train_transform
)

# COCO dataset
coco_train = datasets.CocoDetection(
    root='./data/coco/train2017',
    annFile='./data/coco/annotations/instances_train2017.json',
    transform=transforms.ToTensor()
)

# Custom dataset from folder structure
custom_dataset = datasets.ImageFolder(
    root='./data/custom',  # Folder with subdirectories for each class
    transform=train_transform
)

# Data loaders
train_loader = DataLoader(
    cifar10_train,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    cifar10_test,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Dataset info
print(f"Training samples: {len(cifar10_train)}")
print(f"Test samples: {len(cifar10_test)}")
print(f"Classes: {cifar10_train.classes}")
print(f"Number of classes: {len(cifar10_train.classes)}")
```

### Pre-trained Models

```python
# Image Classification Models
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

# Vision Transformers
vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
vit_l_16 = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)

# EfficientNet models
efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

# ConvNext models
convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
convnext_base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

# Object Detection Models
fasterrcnn_resnet50 = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)

# Semantic Segmentation Models
deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(
    weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)

# Set models to evaluation mode
resnet50.eval()

# Modify models for different number of classes
num_classes = 10  # CIFAR-10 has 10 classes
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

print(f"Model: {resnet50.__class__.__name__}")
print(f"Number of parameters: {sum(p.numel() for p in resnet50.parameters()):,}")
```

## Common Use Cases

### Image Classification

```python
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

class ImageClassifier:
    def __init__(self, num_classes=10, model_name='resnet50', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=pretrained)
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        return avg_loss, accuracy
    
    def predict(self, image_path, transform, class_names):
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return class_names[predicted_class], confidence, probabilities

# Usage example
classifier = ImageClassifier(num_classes=10, model_name='resnet50')

# Training loop
for epoch in range(10):
    train_loss, train_acc = classifier.train_epoch(train_loader)
    test_loss, test_acc = classifier.evaluate(test_loader)
    classifier.scheduler.step()
    
    print(f'Epoch {epoch+1}/10:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Single image prediction
class_names = cifar10_train.classes
prediction, confidence, probs = classifier.predict(
    'path/to/image.jpg', 
    val_transform, 
    class_names
)
print(f"Prediction: {prediction} (Confidence: {confidence:.3f})")
```

### Transfer Learning

```python
def create_transfer_learning_model(num_classes, freeze_features=True):
    """Create a transfer learning model from pre-trained ResNet"""
    
    # Load pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze feature extraction layers (optional)
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

def fine_tune_model(model, train_loader, val_loader, num_epochs=25):
    """Fine-tune a transfer learning model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Different learning rates for different parts
    feature_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            feature_params.append(param)
    
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': feature_params, 'lr': 1e-4},
        {'params': classifier_params, 'lr': 1e-3}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

# Usage
model = create_transfer_learning_model(num_classes=10, freeze_features=False)
model, metrics = fine_tune_model(model, train_loader, test_loader)
```

### Object Detection

```python
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class ObjectDetector:
    def __init__(self, num_classes=91):  # COCO has 80 classes + background
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace classifier head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model.to(self.device)
        
        # COCO class names (for visualization)
        self.coco_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def predict(self, image_path, threshold=0.5):
        """Perform object detection on an image"""
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Filter predictions by confidence threshold
        boxes = predictions[0]['boxes'][predictions[0]['scores'] > threshold]
        labels = predictions[0]['labels'][predictions[0]['scores'] > threshold]
        scores = predictions[0]['scores'][predictions[0]['scores'] > threshold]
        
        results = []
        for box, label, score in zip(boxes, labels, scores):
            results.append({
                'box': box.cpu().numpy(),
                'label': self.coco_names[label.item()],
                'score': score.item()
            })
        
        return results
    
    def visualize_predictions(self, image_path, predictions, save_path=None):
        """Visualize object detection results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        image = Image.open(image_path)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        for pred in predictions:
            box = pred['box']
            label = pred['label']
            score = pred['score']
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            
            ax.add_patch(rect)
            ax.text(
                box[0], box[1] - 10,
                f'{label}: {score:.2f}',
                fontsize=12,
                color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

# Usage
detector = ObjectDetector()
predictions = detector.predict('path/to/image.jpg', threshold=0.5)

print(f"Found {len(predictions)} objects:")
for pred in predictions:
    print(f"  {pred['label']}: {pred['score']:.3f}")

detector.visualize_predictions('path/to/image.jpg', predictions)
```

### Image Segmentation

```python
from torchvision.models.segmentation import deeplabv3_resnet50

class SemanticSegmentation:
    def __init__(self, num_classes=21):  # PASCAL VOC has 20 classes + background
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained DeepLabV3 model
        self.model = deeplabv3_resnet50(pretrained=True)
        
        # Replace classifier for custom number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self.model.to(self.device)
        
        # PASCAL VOC class names
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
            'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'
        ]
        
        # Color palette for visualization
        self.palette = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ]
    
    def segment(self, image_path):
        """Perform semantic segmentation on an image"""
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            output_predictions = output.argmax(1)
        
        # Convert to numpy
        mask = output_predictions.squeeze().cpu().numpy()
        
        return image, mask
    
    def visualize_segmentation(self, image, mask, alpha=0.7):
        """Visualize segmentation results"""
        import matplotlib.pyplot as plt
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(len(self.class_names)):
            colored_mask[mask == class_id] = self.palette[class_id]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(colored_mask, alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_statistics(self, mask):
        """Get statistics about segmented classes"""
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        stats = []
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(self.class_names):
                percentage = (count / total_pixels) * 100
                stats.append({
                    'class': self.class_names[class_id],
                    'pixels': count,
                    'percentage': percentage
                })
        
        return sorted(stats, key=lambda x: x['percentage'], reverse=True)

# Usage
segmenter = SemanticSegmentation()
image, mask = segmenter.segment('path/to/image.jpg')

# Visualize results
segmenter.visualize_segmentation(image, mask)

# Get statistics
stats = segmenter.get_class_statistics(mask)
print("Class distribution:")
for stat in stats:
    if stat['percentage'] > 1.0:  # Only show classes with >1% coverage
        print(f"  {stat['class']}: {stat['percentage']:.1f}%")
```

## Advanced Features

### Custom Datasets

```python
import os
import json
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_paths = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Get label from annotations
        label = self.annotations[self.image_paths[idx]]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CocoStyleDataset(Dataset):
    """Dataset for COCO-style object detection annotations"""
    
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create mappings
        self.image_id_to_path = {img['id']: img['file_name'] for img in self.coco['images']}
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.image_id_to_path.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, self.image_id_to_path[img_id])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(img_id)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# Data loading utilities
def create_data_loaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create data loaders with proper collate function for object detection"""
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Usage
train_dataset = CustomImageDataset(
    root_dir='./data/train',
    annotation_file='./data/train_annotations.json',
    transform=train_transform
)

val_dataset = CustomImageDataset(
    root_dir='./data/val',
    annotation_file='./data/val_annotations.json',
    transform=val_transform
)

train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)
```

### Model Interpretability

```python
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam

def visualize_gradcam(model, image_path, target_class, transform):
    """Visualize GradCAM for a specific class"""
    import matplotlib.pyplot as plt
    import cv2
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Create GradCAM
    target_layer = model.layer4[-1].conv2  # ResNet50 example
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam.numpy(), (image.width, image.height))
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('GradCAM')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Feature visualization
def visualize_feature_maps(model, image_path, layer_name, transform, max_channels=16):
    """Visualize feature maps from a specific layer"""
    
    # Extract features
    feature_extractor = create_feature_extractor(model, return_nodes=[layer_name])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Get features
    with torch.no_grad():
        features = feature_extractor(input_tensor)[layer_name]
    
    # Visualize feature maps
    features = features[0]  # Remove batch dimension
    n_channels = min(max_channels, features.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(n_channels):
        feature_map = features[i].cpu().numpy()
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_channels, 16):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
model = models.resnet50(pretrained=True)
visualize_gradcam(model, 'path/to/image.jpg', target_class=281, transform=val_transform)  # 281 is 'tabby cat' in ImageNet
visualize_feature_maps(model, 'path/to/image.jpg', 'layer4.0.conv1', val_transform)
```

## Integration with Other Libraries

### With OpenCV

```python
import cv2
import numpy as np

def opencv_to_tensor(cv_image):
    """Convert OpenCV image to PyTorch tensor"""
    # OpenCV uses BGR, convert to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image then apply transforms
    pil_image = Image.fromarray(rgb_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_image).unsqueeze(0)

def tensor_to_opencv(tensor):
    """Convert PyTorch tensor to OpenCV image"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp values and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.squeeze().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Real-time inference with webcam
def real_time_classification(model_path, class_names):
    # Load model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Classify frame
        input_tensor = opencv_to_tensor(frame)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        # Draw prediction on frame
        text = f"{class_names[predicted]}: {confidence:.3f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### With Matplotlib for Visualization

```python
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_batch(data_loader, class_names, num_images=8):
    """Visualize a batch of images with labels"""
    
    # Get a batch
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Create grid
    grid = make_grid(images[:num_images], nrow=4, normalize=True, 
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Convert to numpy
    npimg = grid.numpy()
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    
    # Add labels
    for i in range(num_images):
        plt.text(i * (npimg.shape[2] // 4) + 20, 20, 
                class_names[labels[i]], 
                fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualize model predictions
def visualize_predictions(model, test_loader, class_names, num_images=8, device='cpu'):
    """Visualize model predictions vs ground truth"""
    
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Create grid
    images = images.cpu()
    grid = make_grid(images, nrow=4, normalize=True,
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    
    # Add predictions and ground truth
    for i in range(num_images):
        pred_name = class_names[predicted[i]]
        true_name = class_names[labels[i]]
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        
        plt.text(i % 4 * (grid.shape[2] // 4) + 10, 
                (i // 4) * (grid.shape[1] // 2) + 30,
                f'True: {true_name}\nPred: {pred_name}',
                fontsize=10, color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_batch(train_loader, cifar10_train.classes)
plot_training_curves(train_losses, val_losses, train_accs, val_accs)
visualize_predictions(model, test_loader, cifar10_test.classes)
```

## Best Practices

### Performance Optimization

```python
# 1. Use appropriate image sizes
# Smaller images for faster training, larger for better accuracy
resize_transforms = {
    'fast': transforms.Resize(224),      # Standard size
    'balanced': transforms.Resize(256),   # Slightly larger
    'quality': transforms.Resize(384)     # High resolution
}

# 2. Efficient data loading
def create_efficient_dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),  # Use available CPUs
        pin_memory=True,                     # Faster GPU transfer
        persistent_workers=True,             # Keep workers alive
        prefetch_factor=2                    # Prefetch batches
    )

# 3. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, optimizer, criterion, device):
    scaler = GradScaler()
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Use autocast for forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 4. Memory-efficient transforms
memory_efficient_transforms = transforms.Compose([
    transforms.Resize(256, antialias=True),      # Use antialiasing for better quality
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Model Selection and Hyperparameters

```python
# Model recommendations by use case
model_recommendations = {
    'fast_inference': {
        'model': 'mobilenet_v3_small',
        'input_size': 224,
        'batch_size': 64
    },
    'balanced': {
        'model': 'resnet50',
        'input_size': 224,
        'batch_size': 32
    },
    'high_accuracy': {
        'model': 'efficientnet_b7',
        'input_size': 600,
        'batch_size': 8
    },
    'transfer_learning': {
        'model': 'resnet50',
        'input_size': 224,
        'freeze_epochs': 5,
        'total_epochs': 20
    }
}

# Hyperparameter suggestions
def get_hyperparameters(dataset_size, num_classes):
    if dataset_size < 1000:
        return {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 1e-4
        }
    elif dataset_size < 10000:
        return {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 30,
            'weight_decay': 1e-3
        }
    else:
        return {
            'learning_rate': 0.1,
            'batch_size': 64,
            'epochs': 100,
            'weight_decay': 1e-4
        }
```

This comprehensive cheat sheet covers the essential aspects of TorchVision for computer vision tasks. The library provides excellent integration with PyTorch, extensive pre-trained models, and powerful data augmentation capabilities, making it ideal for both research and production computer vision applications.