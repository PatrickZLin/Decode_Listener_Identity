import numpy as np
import os
import torch
import torch.nn as nn
from models import Spiking_vit_MetaFormer
from functools import partial
from spikingjelly.clock_driven import functional
from collections import Counter
import pandas as pd


def segment_test_data(file_path, segment_length=1280):
    data = np.load(file_path)
    num_timesteps = data.shape[1]
    num_segments = num_timesteps // segment_length
    truncated_data = data[:, :num_segments * segment_length]
    
    segments = np.split(truncated_data, num_segments, axis=1)
    return np.array(segments)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Spiking_vit_MetaFormer(
        img_size_h=32,
        img_size_w=1280,
        patch_size=16,
        # embed_dim=[96, 192, 384, 480],
        # embed_dim=[64, 128, 256, 360],
        embed_dim=[32, 64, 128, 192],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=26,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
    ).to(device)

checkpoint = torch.load("YOUR_CKPT_PATH", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

def pad_segments(segments, batch_size):
    num_segments = len(segments)
    padding_needed = batch_size - num_segments
    padding = np.zeros((padding_needed, *segments.shape[1:]))
    segments = np.vstack([segments, padding])
    return segments, padding_needed

def predict_segments(model, segments):
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            padding_needed = 0
            if len(batch) < batch_size:
                batch, padding_needed = pad_segments(batch, batch_size)
            segments_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1)
            segments_tensor = segments_tensor.repeat(1, 3, 1, 1)
            segments_tensor = segments_tensor.to(device)
            predictions = model(segments_tensor)
            predicted_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy()[:batch_size - padding_needed])

            functional.reset_net(model)

    return predicted_labels

def majority_vote(predicted_labels):
    label_counts = Counter(predicted_labels)
    final_label = max(label_counts, key=label_counts.get)
    return final_label

def normalize_eeg(data, axis=(1, 2)):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std)

def test_model_on_folder(model, test_folder, segment_length=1280):
    predictions = {}
    
    for file_name in os.listdir(test_folder):
        if file_name.endswith("_eeg.npy"):
            id_only = file_name.split('_')[0]
            file_path = os.path.join(test_folder, file_name)
            
            segments = segment_test_data(file_path, segment_length)
            segments = normalize_eeg(segments, axis=(1, 2))
            segment_labels = predict_segments(model, segments)
            final_label = majority_vote(segment_labels)
            predictions[id_only] = final_label
    
    return predictions

test_folder = "YOUR_PREPROCESSED_DATASET_PATH"
predictions = test_model_on_folder(model, test_folder)

ground_truth = pd.read_csv('results_subject_identification.csv')
predictions_df = pd.DataFrame(list(predictions.items()), columns=['id', 'predicted_label'])
ground_truth['id'] = ground_truth['id'].astype(str)
predictions_df['id'] = predictions_df['id'].astype(str)
comparison = pd.merge(ground_truth, predictions_df, on='id')
comparison['correct'] = comparison['prediction'] == comparison['predicted_label']
accuracy = comparison['correct'].mean()
print(f"Accuracy: {accuracy:.2%}")