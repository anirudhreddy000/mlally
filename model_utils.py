import torch
from torch import nn, optim
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import io

os.makedirs("heatmaps", exist_ok=True)

def fine_tune_clip(image_cat: Image.Image, image_dog: Image.Image, num_epochs=10):
    """Fine-tune CLIP model on cat and dog images"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.train()

    texts = ["a cat", "a dog"]
    images = [image_cat, image_dog]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    training_results = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        labels = torch.arange(len(images)).to(device)

        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        loss.backward()
        optimizer.step()
        
        training_results.append(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")
        print(training_results[-1])

    model.save_pretrained("clip-cat-dog")
    processor.save_pretrained("clip-cat-dog")
    
    return training_results

def generate_attention_heatmap(image: Image.Image, attention_map):
    """Generate a heatmap from attention weights and save to file"""
    attn = attention_map.squeeze(0).mean(0)  
    cls_attention = attn[0, 1:]  
    
    num_image_tokens = cls_attention.shape[0]
    grid_size = int(np.sqrt(num_image_tokens))
    
    if grid_size * grid_size != num_image_tokens:
        for i in range(int(np.sqrt(num_image_tokens)), 0, -1):
            if num_image_tokens % i == 0:
                grid_h, grid_w = i, num_image_tokens // i
                break
        else:
            grid_h = grid_w = grid_size
    else:
        grid_h = grid_w = grid_size
    
    if grid_h * grid_w > num_image_tokens:
        pad_size = grid_h * grid_w - num_image_tokens
        cls_attention = torch.cat([cls_attention, torch.zeros(pad_size, device=cls_attention.device)])
    elif grid_h * grid_w < num_image_tokens:
        cls_attention = cls_attention[:grid_h * grid_w]
    
    attention_grid = cls_attention.reshape(grid_h, grid_w).cpu().numpy()
    
    attention_grid = attention_grid - attention_grid.min()
    max_val = attention_grid.max()
    if max_val > 0:
        attention_grid = attention_grid / max_val
    
    heatmap = Image.fromarray((attention_grid * 255).astype(np.uint8)).resize(
        image.size, resample=Image.BILINEAR
    )
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Attention Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    filename = f"heatmap_{uuid.uuid4()}.png"
    filepath = f"heatmaps/{filename}"
    
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def predict_class(image: Image.Image):
    """Predict class of image using fine-tuned CLIP model and generate attention heatmap"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("clip-cat-dog", attn_implementation="eager").to(device)
    processor = CLIPProcessor.from_pretrained("clip-cat-dog")
    model.eval()
    
    texts = ["a cat", "a dog"]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        predicted_idx = probs.argmax().item()
        predicted = texts[predicted_idx]
        confidence = float(probs[predicted_idx].item())  
        
        # Extract vision model attention
        vision_attentions = outputs.vision_model_output.attentions
        
        if vision_attentions and len(vision_attentions) > 0:
            # Use the last layer's attention
            last_layer_attention = vision_attentions[-1]
            heatmap_path = generate_attention_heatmap(image, last_layer_attention)
        else:
            # Create a dummy heatmap if no attention is available
            heatmap_path = "heatmaps/no_attention.png"
            if not os.path.exists(heatmap_path):
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.title("No Attention Available")
                plt.axis('off')
                plt.savefig(heatmap_path)
                plt.close()
    
    return predicted, confidence, heatmap_path