"""
Основной pipeline для 3D morphing между source и target изображениями
"""
import torch
import numpy as np
from typing import List, Tuple, Union
from PIL import Image

from encoder_dinov2 import DINOv2Encoder
from barycenter_optimization import BarycenterOptimizer
from trellis_decoder import TrellisDecoder


class MorphingPipeline:
    """
    Основной pipeline для textured 3D morphing
    Реализует: DINOv2 encoder -> Barycenter optimization -> Trellis1 decoder
    """
    
    def __init__(
        self,
        dino_model: str = "dinov2_vitl14",
        barycenter_reg: float = 0.1,
        device: str = "cuda"
    ):
        """
        Инициализация pipeline
        
        Args:
            dino_model: модель DINOv2 для encoder
            barycenter_reg: параметр регуляризации для barycenter
            device: устройство для вычислений
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Инициализация компонентов
        print("Loading DINOv2 encoder...")
        self.encoder = DINOv2Encoder(model_name=dino_model, device=self.device)
        
        print("Initializing barycenter optimizer...")
        self.barycenter_opt = BarycenterOptimizer(reg=barycenter_reg)
        
        print("Loading Trellis decoder...")
        self.decoder = TrellisDecoder(device=self.device)
    
    def morph(
        self,
        source_image: Union[str, Image.Image],
        target_image: Union[str, Image.Image],
        num_steps: int = 10,
        reduce_tokens: bool = False,
        n_clusters: int = 256
    ) -> List[Tuple[dict, dict]]:
        """
        Выполнение morphing между source и target изображениями
        
        Args:
            source_image: путь к source изображению или PIL Image
            target_image: путь к target изображению или PIL Image
            num_steps: число шагов интерполяции
            reduce_tokens: уменьшать ли число токенов через кластеризацию
            n_clusters: число кластеров для уменьшения токенов
            
        Returns:
            results: список кортежей (mesh, texture) для каждого шага морфинга
        """
        # Загрузка изображений
        if isinstance(source_image, str):
            source_img = Image.open(source_image).convert('RGB')
        else:
            source_img = source_image
        
        if isinstance(target_image, str):
            target_img = Image.open(target_image).convert('RGB')
        else:
            target_img = target_image
        
        # Шаг 1: Кодирование изображений через DINOv2
        print("Encoding source image...")
        lat_src = self.encoder.encode(source_img)
        
        print("Encoding target image...")
        lat_tgt = self.encoder.encode(target_img)
        
        # Шаг 2: Вычисление morphing latents через barycenter optimization
        print(f"Computing barycenter sequence ({num_steps} steps)...")
        morphing_latents = self.barycenter_opt.compute_morphing_sequence(
            lat_src.cpu().numpy(),
            lat_tgt.cpu().numpy(),
            num_steps=num_steps,
            reduce_tokens=reduce_tokens,
            n_clusters=n_clusters
        )
        
        # Шаг 3: Декодирование каждого латента в 3D через Trellis
        print("Decoding latents to 3D...")
        results = []
        
        for i, lat_t in enumerate(morphing_latents):
            print(f"  Decoding step {i+1}/{len(morphing_latents)}...")
            outputs = self.decoder.decode(lat_t)
            results.append(outputs)
        
        print("Morphing completed!")
        return results
