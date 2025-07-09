import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def print_metrics_and_confusion_matrix(all_labels, all_preds, class_names):
    # Calculate metrics
    print("Classification Report:")
    classification_rep = classification_report(all_labels, all_preds, target_names=class_names)
    print(classification_rep)

    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations for test dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Specify the data directory and create a data loader for test
    data_dir = 'D:/EDI_code/Images/dataset 2'
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load your pre-trained model
    model = models.resnet50(pretrained=False)  # Assuming you're using ResNet-50 architecture
    num_classes = len(test_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('resnet50_model_finetuned.pth'))  # Replace 'your_model.pth' with the actual file path
    model = model.to(device)

    # Evaluate the model on the test dataset
    all_labels, all_preds = evaluate_model(model, test_loader, device)

    # Print metrics and confusion matrix for each class
    print_metrics_and_confusion_matrix(all_labels, all_preds, test_dataset.classes)


if __name__ == '__main__':
    # Call the main function
    main()
