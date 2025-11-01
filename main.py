import os
import seaborn as sns
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from LeNet_model import LeNetUpdated
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_deterministic(42)

# Accelerator
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print("Device: ", device)

# Hyperparameters
learning_rate = 3e-4
num_epochs = 60
batch_size = 32
num_classes = 10


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transforms
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transforms
)
# Dataloaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("Length of training data: ", len(training_data))
print("Length of test data: ", len(test_data))

# model
model = LeNetUpdated().to(device)
print(model)
model.apply(init_weights)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

best_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward pass
        outputs = model(imgs)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, one_hot)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)

    train_loss = running / len(training_data)
    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {running / len(training_data):.4f}")
    if train_loss < best_loss:
        best_loss = train_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        print("Best model saved")

model.load_state_dict(best_model_state)
model.eval()
total = 0
correct = 0
test_loss = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for imgs, labels in test_dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, one_hot)
        test_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

all_labels_np = np.array(all_labels)
all_predictions_np = np.array(all_predictions)

misclassified_mask = all_predictions_np != all_labels_np
num_misclassified = np.sum(misclassified_mask)
print(f"Number of misclassified examples: {num_misclassified}")

test_loss /= len(test_dataloader)
test_acc = (correct / total) * 100

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig('LeNet_confusion_matrix.png', bbox_inches='tight')
plt.show()
