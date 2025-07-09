import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations for training dataset
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Specify the data directory and create a data loader for training
    data_dir = 'D:/EDI_code/Images/dataset 2'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Fine-tune: Unfreeze certain layers for training
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers by default
    for param in model.layer4.parameters():
        param.requires_grad = True  # Unfreeze the last residual block for fine-tuning

    num_classes = 5
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Send the model to the GPU if available
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 5

    train_losses = []  # List to store train losses

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)  # Append train loss to the list

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f}')

    # Save the trained model weights
    torch.save(model.state_dict(), 'resnet50_model_finetuned.pth')
    print("Training complete! Model saved as 'resnet50_model_finetuned.pth'")

    # Plot the loss curve
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.show()


if __name__ == '__main__':
    # Call the main function
    main()



# #  updated one
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms, datasets
# from torch.utils.data import DataLoader
#
# def main():
#     # Set the device (CPU or GPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Define data transformations for training dataset
#     data_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # Specify the data directory and create a data loader for training
#     data_dir = 'D:/EDI_code/Images/dataset 2'
#     train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#
#     # Load the pre-trained ResNet-50 model
#     model = models.resnet50(pretrained=True)
#
#     # Fine-tune: Unfreeze certain layers for training
#     for param in model.parameters():
#         param.requires_grad = False  # Freeze all layers by default
#     for param in model.layer4.parameters():
#         param.requires_grad = True  # Unfreeze the last residual block for fine-tuning
#
#     num_classes = 5
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#
#     # Send the model to the GPU if available
#     model = model.to(device)
#
#     # Define the loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
#
#     # Training loop
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#
#         epoch_loss = running_loss / len(train_dataset)
#         print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f}')
#
#     # Save the trained model weights
#     torch.save(model.state_dict(), 'resnet50_model_finetuned.pth')
#     print("Training complete! Model saved as 'resnet50_model_finetuned.pth'")
#
# if __name__ == '__main__':
#     # Call the main function
#     main()
#

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
#
# def main():
#     # Set the device (CPU or GPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Define data transformations for the training dataset
#     data_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # Specify the data directory and create a data loader for training
#     data_dir = 'D:/EDI_code/Images/dataset 2'
#     train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#
#     # Load the pre-trained ResNet-50 model
#     model = models.resnet50(pretrained=True)
#     num_classes = 5
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#
#     # Send the model to the GPU if available
#     model = model.to(device)
#
#     # Define the loss function
#     criterion = nn.CrossEntropyLoss()
#
#     # Experiment with different learning rates and optimizers
#     learning_rates = [0.001, 0.0001]  # You can add more learning rates to test
#     optimizers = [optim.SGD, optim.Adam]  # You can try different optimizers
#
#     for lr in learning_rates:
#         for optimizer_class in optimizers:
#             # Create the optimizer with the selected learning rate
#             optimizer = optimizer_class(model.parameters(), lr=lr)
#
#             # Training loop
#             num_epochs = 10
#             for epoch in range(num_epochs):
#                 model.train()
#                 running_loss = 0.0
#                 for inputs, labels in train_loader:
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)
#                     optimizer.zero_grad()
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()
#                     running_loss += loss.item() * inputs.size(0)
#
#                 epoch_loss = running_loss / len(train_dataset)
#                 print(f'LR: {lr} | Optimizer: {optimizer_class.__name__} | Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f}')
#
#             # Save the trained model weights for each configuration
#             model_name = f'resnet50_model_lr_{lr}{optimizer_class.__name__}.pth'
#             torch.save(model.state_dict(), model_name)
#             print(f"Training complete! Model saved as '{model_name}'")
#
# if __name__ == '__main__':
#    # Call the main function
#    main()