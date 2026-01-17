"""
Основной pipeline для 3D morphing между source и target изображениями
"""
import torch
import numpy as np
from typing import List, Tuple, Union
from PIL import Image

from encoder_dinov2 import DINOv2Encoder
from barycenter_optimization import BarycenterOptimizer
from trellis_decoder import TrellisDecoder, PlaceholderTrellisDecoder


class MorphingPipeline:
    """
    Основной pipeline для textured 3D morphing
    Реализует: DINOv2 encoder -> Barycenter optimization -> Trellis1 decoder
    """
    
    def __init__(
        self,
        dino_model: str = "dinov2_vitb14",
        barycenter_reg: float = 0.1,
        device: str = "cuda",
        use_placeholder_decoder: bool = False
    ):
        """
        Инициализация pipeline
        
        Args:
            dino_model: модель DINOv2 для encoder
            barycenter_reg: параметр регуляризации для barycenter
            device: устройство для вычислений
            use_placeholder_decoder: использовать placeholder decoder (если Trellis не установлен)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Инициализация компонентов
        print("Loading DINOv2 encoder...")
        self.encoder = DINOv2Encoder(model_name=dino_model, device=self.device)
        
        print("Initializing barycenter optimizer...")
        self.barycenter_opt = BarycenterOptimizer(reg=barycenter_reg)
        
        print("Loading Trellis decoder...")
        if use_placeholder_decoder:
            self.decoder = PlaceholderTrellisDecoder(device=self.device)
        else:
            try:
                self.decoder = TrellisDecoder(device=self.device)
                # Проверяем, загружена ли модель
                if self.decoder.pipeline is None and self.decoder.mesh_decoder is None:
                    print("Trellis model not loaded, using placeholder...")
                    self.decoder = PlaceholderTrellisDecoder(device=self.device)
                else:
                    print("Trellis decoder loaded successfully!")
            except Exception as e:
                print(f"Failed to load Trellis decoder: {e}")
                print("Using placeholder decoder...")
                self.decoder = PlaceholderTrellisDecoder(device=self.device)
    
    def morph(
        self,
        source_image: Union[str, Image.Image],
        target_image: Union[str, Image.Image],
        num_steps: int = 10,
        reduce_tokens: bool = True,
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
            mesh, texture = self.decoder.decode(lat_t)
            results.append((mesh, texture))
        
        print("Morphing completed!")
        return results
    
    def morph_step(
        self,
        source_image: Union[str, Image.Image],
        target_image: Union[str, Image.Image],
        alpha: float,
        reduce_tokens: bool = True,
        n_clusters: int = 256
    ) -> Tuple[dict, dict]:
        """
        Вычисление одного шага morphing для заданного alpha
        
        Args:
            source_image: source изображение
            target_image: target изображение
            alpha: интерполяционный параметр [0, 1]
            reduce_tokens: уменьшать ли число токенов
            n_clusters: число кластеров
            
        Returns:
            mesh, texture: mesh и texture для заданного alpha
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
        
        # Кодирование
        lat_src = self.encoder.encode(source_img)
        lat_tgt = self.encoder.encode(target_img)
        
        # Barycenter для одного alpha
        latents_src_np = lat_src.cpu().numpy()
        latents_tgt_np = lat_tgt.cpu().numpy()
        
        if reduce_tokens:
            latents_src_np, weights_src = self.barycenter_opt.reduce_tokens(latents_src_np, n_clusters)
            latents_tgt_np, weights_tgt = self.barycenter_opt.reduce_tokens(latents_tgt_np, n_clusters)
        else:
            weights_src = None
            weights_tgt = None
        
        barycenter_latent = self.barycenter_opt.compute_barycenter(
            latents_src_np,
            latents_tgt_np,
            alpha,
            weights_src=weights_src,
            weights_tgt=weights_tgt
        )
        
        # Декодирование
        mesh, texture = self.decoder.decode(barycenter_latent)
        
        return mesh, texture