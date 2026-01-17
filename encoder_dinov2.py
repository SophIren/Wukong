"""
DINOv2 Encoder для кодирования source и target изображений
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, Tuple
import numpy as np


class DINOv2Encoder:
    """DINOv2 encoder для извлечения латентных представлений из изображений"""
    
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        """
        Инициализация DINOv2 encoder
        
        Args:
            model_name: имя модели DINOv2 ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Загрузка модели DINOv2
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception as e:
            # Альтернативный способ загрузки
            import timm
            self.model = timm.create_model(model_name, pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Препроцессинг изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def encode(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Кодирование изображения в латентное представление
        
        Args:
            image: PIL Image или torch.Tensor [C, H, W]
            
        Returns:
            latents: тензор с латентами [num_patches, dim] или [dim] для global token
        """
        if isinstance(image, Image.Image):
            image = self.transform(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # Получаем patch embeddings и cls token
            # DINOv2 возвращает словарь с 'x_norm_clstoken' и 'x_norm_patchtokens'
            features = self.model.get_intermediate_layers(image, n=1)[0]
            
            # features shape: [batch, num_patches + 1, dim] (включая cls token)
            # Разделяем cls token и patch tokens
            if features.dim() == 3:
                # Извлекаем patch tokens (без cls token)
                patch_tokens = features[:, 1:, :]  # [1, num_patches, dim]
                patch_tokens = patch_tokens.squeeze(0)  # [num_patches, dim]
            else:
                patch_tokens = features
        
        return patch_tokens
