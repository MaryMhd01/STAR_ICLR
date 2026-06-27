#########################
# only use it to generate attention heatmap
##################first step

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import requests
import time
import numpy as np
import cv2
import copy
import sys
import os
from torch.utils.data import Dataset, random_split, DataLoader

np.set_printoptions(threshold=sys.maxsize)

# ====================== CLS2IDX (ImageNet Classes) ======================
try:
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes = requests.get(url, timeout=10).text.strip().split("\n")
    CLS2IDX = {i: cls for i, cls in enumerate(classes)}
    print(f"✅ Successfully loaded {len(CLS2IDX)} ImageNet classes")
except Exception as e:
    print(f"❌ Failed to load CLS2IDX: {e}")
    CLS2IDX = None
# =====================================================================

import math
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as vit_base
from baselines.ViT.ViT_LRP import deit_small_patch16_224 as vit_small
from baselines.ViT.ViT_LRP import deit_tiny_patch16_224 as vit_tiny
from baselines.ViT.ViT_LRP import Block
from baselines.ViT.ViT_explanation_generator import LRP

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained with DeiT from Google Drive

model_path = "/content/drive/MyDrive/STAR/pretrained_models/DeiT-Tiny.pth"

model = vit_tiny(pretrained=False).cuda()
model.load_state_dict(torch.load(model_path, map_location="cuda"))
model.eval()
attribution_generator = LRP(model)
print("✅ DeiT-Tiny model loaded successfully!")

def print_top_classes(predictions, **kwargs):
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
   
    print('Top 5 classes:')
    max_str_len = 0
    class_names = []
   
    for cls_idx in class_indices:
        if CLS2IDX is not None:
            full_name = CLS2IDX[cls_idx]
            class_name = full_name.split(',')[0].strip()
            class_names.append(class_name)
            max_str_len = max(max_str_len, len(full_name))
        else:
            class_names.append(f"Class_{cls_idx}")
   
    for i, cls_idx in enumerate(class_indices):
        if CLS2IDX is not None:
            output_string = '\t{} : {}'.format(cls_idx, class_names[i])
            output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
            output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(
                predictions[0, cls_idx], 100 * prob[0, cls_idx]
            )
        else:
            output_string = '\t{} : Class_{}'.format(cls_idx, cls_idx)
        print(output_string)
   
    return class_indices

def add_visualization(original_image, class_index=None, start_layer=None):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(), 
        method="transformer_attribution", 
        index = class_index, 
        start_layer=start_layer
    ).detach()
    
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear'
    )
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution

def generate_visualization(original_image, class_index=None, start_layer=None):
    i = 0
    transformer_attribution = None
    for image in original_image:
        for index in class_index:
            temp = add_visualization(image, class_index=index, start_layer=start_layer)
            if transformer_attribution is None:
                transformer_attribution = temp / len(original_image)
            else:
                transformer_attribution += temp / len(original_image)
        i += 1
    return transformer_attribution
# ====================== Custom ImageNet Dataset for Colab ======================
class CustomImageNetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        label = 0  # dummy label
        if self.transform:
            img = self.transform(img)
        return img, label, img_path

# ====================== Data Loading ======================
image_dir = '/content/imagenet_val'   

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageNetDataset(image_dir, transform=transform)
test_dataset, _ = random_split(dataset, [10000, len(dataset) - 10000])

train_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, 
                         num_workers=2, pin_memory=True)

print(f"✅ Custom DataLoader ready with {len(test_dataset)} images")
# =====================================================================
def genr_decision(model, train_loader, num_batches=30):
    transformer_attribution = [torch.zeros(224, 224) for _ in range(12)]
   
    print(f"Starting attention map generation for {num_batches} batches...")
   
    processed = 0
    for i, batch in enumerate(train_loader):
        if processed >= num_batches:
            break
           
        try:
            images = batch[0].cuda()
            output = model(images)
            class_top5 = print_top_classes(output)
           
            print(f"Batch {i+1} | Shape: {images.shape} | Processed: {processed+1}/{num_batches}")
           
            for layer in range(12):
                temp = generate_visualization(images, class_index=class_top5, start_layer=layer)
                if isinstance(temp, torch.Tensor):
                    temp = temp.cpu().numpy()
                transformer_attribution[layer] += temp / num_batches
               
            processed += 1
            del images, output
            torch.cuda.empty_cache()
           
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory - skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise exception
                
    print(f"✅ Finished! {processed} batches processed successfully.")
    return transformer_attribution

# ====================== Generate Attention Maps ======================
print("🚀 Starting final attention map generation...")
attention_map = genr_decision(model, train_loader, num_batches=30)

#save_dir = "/content/recordattn_base"
#os.makedirs(save_dir, exist_ok=True)
save_dir = "/content/drive/MyDrive/STAR/recordattn_base"
os.makedirs(save_dir, exist_ok=True)

for l in range(12):
    print(f"layer: {l}")
    np.save(f"{save_dir}/layer192_{l}.npy", attention_map[l].cpu().numpy())

print("✅ All layers saved successfully!")
# =====================================================================
