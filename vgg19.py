# model training
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Testing': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/brain-tumor-dataset/' 

image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', data_transforms[x]) for x in ['Training', 'Validation', 'Testing']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2) for x in ['Training', 'Validation', 'Testing']}

class_names = image_datasets['Training'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
print(f'Classes: {class_names}')

model = models.vgg19(weights='VGG19_Weights.DEFAULT')

for param in model.features.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)
print('VGG-19 model is ready.')


import torch.optim as optim
import time
import copy
from sklearn.metrics import precision_score, recall_score, f1_score

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_eval_with_save(model, dataloaders, criterion, optimizer, num_epochs=50, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict()) 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
            epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
                  f'Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} '
                  f'F1-Score: {epoch_f1:.4f}')

        # Early Stopping and saving the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict()) # Save the best model weights
            print(f'Saving new best model with loss: {best_loss:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                model.load_state_dict(best_model_wts) # Load the best weights back
                return model

    print('Training Complete.')
    model.load_state_dict(best_model_wts) # Load best weights at the end
    return model

# Start the training process
model_trained = train_and_eval_with_save(model, dataloaders, criterion, optimizer, num_epochs=50, patience=5)

# model evaluation

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# ----------------- Part 1: Setup and Model Definition -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.vgg19(weights=None)
for param in model.features.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
num_classes = 4
model.classifier[6] = nn.Linear(num_ftrs, num_classes)
model.to(device)

weights_path = 'D:\\hackothan\\best_vgg19_model.pth' 
model.load_state_dict(torch.load(weights_path))
print("Model weights loaded successfully!")

# ----------------- Part 2: Data Loading and Evaluation -----------------
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = r'D:\\hackothan\\brain-tumor-dataset'
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Testing'), data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
class_names = test_dataset.classes

def get_test_metrics(model, test_dataloader, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Classification Report 
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df_report.values, colLabels=df_report.columns, rowLabels=df_report.index, cellLoc='center', loc='center')
    plt.title('Classification Report')
    plt.savefig('classification_report.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("Classification Report saved to classification_report.png")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    # Overall Accuracy
    accuracy = np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    get_test_metrics(model, test_dataloader, class_names)

# streamlit interface 
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

#  Model Setup 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights=None)
for param in model.features.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
num_classes = 4
model.classifier[6] = nn.Linear(num_ftrs, num_classes)
model.to(device)
weights_path = 'D:\\hackothan\\best_vgg19_model.pth' 
model.load_state_dict(torch.load(weights_path))
model.eval()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# UI 
st.set_page_config(layout="centered", initial_sidebar_state="auto")

st.markdown("""
<style>
.stApp {
    background-color: #121212; /* Dark, elegant background */
    color: #e0e0e0; /* Light grey for text */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, h4 {
    color: #bb86fc; /* A beautiful, soft purple for headings */
}
.stFileUploader label {
    font-size: 1.2rem;
    font-weight: bold;
    color: #e0e0e0;
}
.stImage {
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.5); /* Stronger shadow for contrast */
    border: 1px solid #424242; /* A subtle border */
}
.stAlert {
    border-radius: 10px;
    font-weight: bold;
}
.st-bs { /* This is for the custom success alert */
    background-color: #388e3c;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}
.st-bw { /* This is for the custom warning alert */
    background-color: #f57f17;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Detection")
st.markdown("### Upload an MRI scan to check for a tumor")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)
    st.markdown("---")
    st.write("#### Classifying...")

    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item() * 100

    st.subheader("Prediction:")
    if predicted_class == 'notumor':
        st.markdown(f'<div class="st-bs">The model predicts: **No Tumor Detected**</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="st-bw">The model predicts: **{predicted_class.title()} Tumor Detected**</div>', unsafe_allow_html=True)

    st.info(f"Confidence: **{confidence:.2f}%**")