import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. SWAP DATASET: ImageFolder cannot handle bounding box labels. You need a custom Dataset.
class BoundingBoxDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        """
        annotations: A dictionary or list mapping image filenames to 
                     normalized bounding boxes [xmin, ymin, xmax, ymax] (0 to 1 scale)
        """
        self.img_dir = img_dir
        self.annotations = annotations
        self.img_names = list(annotations.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Bounding box coordinates normalized between 0 and 1
        bbox = torch.tensor(self.annotations[img_name], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, bbox

# 2. ADAPT VISUALIZATION: Update to draw rectangles instead of printing text classes
def visualize_test_predictions(model, test_loader, num_images=6):
    model.eval()
    images, true_boxes = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        pred_boxes = model(images)
    
    images = images.cpu()
    true_boxes = true_boxes.cpu().numpy()
    pred_boxes = pred_boxes.cpu().numpy()
    
    fig, axes = plt.subplots(2, max(1, num_images // 2), figsize=(15, 10))
    axes = axes.flatten()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for idx in range(min(num_images, len(images))):
        ax = axes[idx]
        img = images[idx].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        
        height, width, _ = img.shape
        
        # Helper to convert normalized [xmin, ymin, xmax, ymax] to matplotlib format
        def draw_box(box, color):
            xmin, ymin, xmax, ymax = box * [width, height, width, height]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
        draw_box(true_boxes[idx], 'green') # Green = Ground Truth
        draw_box(pred_boxes[idx], 'red')   # Red = Prediction
        ax.set_title("Green: True, Red: Pred")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- Dummy data simulation ---
    # Replace this block with your actual bounding box coordinates parsing logic (e.g., from a CSV or JSON file)
    img_dir = '/Users/visitor/Documents/Hackgwinnett26/data'
    mock_annotations = {}
    if os.path.exists(img_dir):
        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_imgs = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    rel_path = os.path.relpath(os.path.join(root, f), img_dir)
                    all_imgs.append(rel_path)
        
        mock_annotations = {img: [0.1, 0.1, 0.9, 0.9] for img in all_imgs}
    else:
        mock_annotations = {"sample.jpg": [0.2, 0.2, 0.8, 0.8]}

    dataset = BoundingBoxDataset(img_dir=img_dir, annotations=mock_annotations, transform=data_transforms)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
        test_loss = running_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
    visualize_test_predictions(model, test_loader, num_images=6)
