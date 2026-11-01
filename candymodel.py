import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split 
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from PIL import Image


if __name__=='__main__':
    """

    from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize all images to 224x224
    transforms.ToTensor(),               # Convert PIL Image to PyTorch Tensor (scales to [0.0, 1.0])
    transforms.Normalize(                # Normalize with ImageNet mean and standard deviation
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Dummy label dictionary maps filename to class index (e.g., {'img1.jpg': 0, 'img2.jpg': 1})
my_labels = {"cat1.jpg": 0, "dog1.jpg": 1, "cat2.jpg": 0} 

# Instantiate Dataset
dataset = CustomImageDataset(
    image_dir="path/to/your/images", 
    labels_dict=my_labels, 
    transform=data_transforms
)

# Create DataLoader
train_loader = DataLoader(
    dataset=dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2
)

    """
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root='/Users/visitor/Documents/Hackgwinnett26/data', transform=data_transforms)


    #dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(3 * 224 * 224, 10)

        def forward(self, x):
            x = x.view(-1, 3 * 224 * 224) 
            x = self.fc(x)
            return x

    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(running_loss / len(test_loader))
        test_accuracies.append(100 * correct / total)

        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, "
            f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    def visualize_test_predictions(model, test_loader, dataset, num_images=6):
        model.eval()
        
        # Get a batch of test images
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        # Get class names from ImageFolder mapping
        class_names = dataset.classes 
        
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
        # Move tensors back to CPU for numpy conversions
        images = images.cpu()
        labels = labels.cpu()
        preds = preds.cpu()
        
        # Plotting grid configuration
        fig = plt.figure(figsize=(15, 10))
        
        for idx in range(min(num_images, len(images))):
            ax = fig.add_subplot(2, max(1, num_images // 2), idx + 1, xticks=[], yticks=[])
            
            # Step 1: Convert from Tensor (C, H, W) to Numpy (H, W, C)
            img = images[idx].permute(1, 2, 0).numpy()
            
            # Step 2: Denormalize using ImageNet mean & std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1) # Ensure values stay inside [0, 1] bounds
            
            plt.imshow(img)
            
            # Determine title color based on prediction accuracy
            is_correct = preds[idx].item() == labels[idx].item()
            color = "green" if is_correct else "red"
            
            ax.set_title(
                f"Pred: {class_names[preds[idx].item()]}\nTrue: {class_names[labels[idx].item()]}",
                color=color
            )
            
        plt.tight_layout()
        plt.show()

    # --- Call this at the very end of your script (after plt.show()) ---
    import numpy as np
    visualize_test_predictions(model, test_loader, dataset, num_images=6)

    # # Plotting the results
    # plt.figure(figsize=(12, 5))

    # # Plot training and validation loss
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, epochs + 1), test_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    # plt.plot(range(1, epochs + 1), test_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
