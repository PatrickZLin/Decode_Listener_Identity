import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vit_pytorch.cct import CCT


LR_SCHEDULE = True

start_epoch = 1

with open('data/splits/splits_subject_identification.json', 'r') as f:
    data = json.load(f)

train_data = data['train']
val_data = data['val_trial']

id_to_subject_train = {f"{item['id']}_eeg.npy": item['subject_id'] for item in train_data}
id_to_subject_val = {f"{item['id']}_eeg.npy": item['subject_id'] for item in val_data}

id_to_subject = id_to_subject_train | id_to_subject_val

def segment_eeg(data, label, segment_length=1280):
    num_timesteps = data.shape[1]
    num_segments = num_timesteps // segment_length
    truncated_data = data[:, :num_segments * segment_length]
    
    segments = np.split(truncated_data, num_segments, axis=1)
    labels = [label] * num_segments

    return segments, labels

def process_eeg_dataset(path_to_eeg_files, id_to_subject, segment_length=1280):
    X, y = [], []
    
    for file_name, subject_id in id_to_subject.items():
        file_path = f"{path_to_eeg_files}/{file_name}"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        eeg_data = np.load(file_path)
        segments, labels = segment_eeg(eeg_data, subject_id, segment_length)
        X.extend(segments)
        y.extend(labels)
    
    return np.array(X), np.array(y)

path_to_eeg_files = "YOUR_PREPROCESSED_DATASET_PATH"

all_data, all_labels = process_eeg_dataset(path_to_eeg_files, id_to_subject, segment_length=1280) 

def normalize_eeg(data, axis=(1, 2)):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std)

all_data = normalize_eeg(all_data)

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

X_train, X_val, y_train, y_val = train_test_split(
    all_data, all_labels_encoded, test_size=0.1, random_state=42, stratify=all_labels_encoded
)

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        sample = sample.repeat(3, 1, 1)
        label = self.labels[idx]
        return sample, torch.tensor(label, dtype=torch.long)
    
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

batch_size = 8 # Set the batch size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CCT(
        img_size = (32, 1280),
        embedding_dim = 384,
        n_conv_layers = 2,
        kernel_size = 7,
        stride = 2,
        padding = 3,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1,
        num_layers = 14,
        num_heads = 6,
        mlp_ratio = 3.,
        num_classes = 26,
        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
    ).to(device)

# model = CCT(
#         img_size = (32, 1280),
#         embedding_dim = 192,
#         n_conv_layers = 2,
#         kernel_size = 7,
#         stride = 2,
#         padding = 3,
#         pooling_kernel_size = 3,
#         pooling_stride = 2,
#         pooling_padding = 1,
#         num_layers = 7,
#         num_heads = 6,
#         mlp_ratio = 3.,
#         num_classes = 26,
#         positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
#     ).to(device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of params (M): %.2f" % (n_parameters / 1.e6))

class_counts = torch.bincount(torch.tensor(all_labels_encoded, dtype=torch.int64))
class_weights = 1.0 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if LR_SCHEDULE:
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',    
        factor=0.5,    
        patience=5,    
        verbose=True,  
        threshold=1e-4,
        cooldown=2,    
        min_lr=1e-6    
    )

def train_one_epoch(model, train_loader, optimizer, criterion):
    global global_step
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

num_epochs = 300

save_path = 'YOUR_CKPT_SAVE_PATH'

best_val_accuracy = 95

for epoch in range(start_epoch, start_epoch + num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f'Epoch [{epoch}/{start_epoch + num_epochs - 1}]')
    print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    if LR_SCHEDULE:
        scheduler.step(val_loss)

    if epoch % 10 == 0:
        save_file = save_path.format(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if LR_SCHEDULE else None,
            'train_loss': train_loss
        }, save_file)
        print(f"Model and optimizer saved to {save_file}")

    if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        save_file = save_path.format('best')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if LR_SCHEDULE else None,
            'train_loss': train_loss
        }, save_file)
        print(f"Model and optimizer saved to {save_file}")