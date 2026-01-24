"""
Основной pipeline для 3D morphing между source и target изображениями
Реализует статью: Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models
"""
import torch
import numpy as np
from typing import List, Tuple, Union
from PIL import Image

from barycenter_optimization import BarycenterOptimizer
from trellis_decoder import TrellisDecoder


class MorphingPipeline:
    """
    Основной pipeline для textured 3D morphing
    Реализует: TRELLIS encode_image -> Barycenter optimization -> Trellis decoder
    """
    
    def __init__(
        self,
        device: str = "cuda"
    ):
        """
        Инициализация pipeline
        
        Args:
            barycenter_reg: параметр регуляризации для barycenter
            device: устройство для вычислений
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Инициализация компонентов
        print("Initializing barycenter optimizer...")
        self.barycenter_opt = BarycenterOptimizer()
        
        print("Loading Trellis decoder...")
        self.decoder = TrellisDecoder(device=self.device)
    
    def morph(
        self,
        source_image: Union[str, Image.Image],
        target_image: Union[str, Image.Image],
        num_steps: int = 10,
        preprocess_image: bool = True
    ) -> List[Tuple[dict, dict]]:
        """
        Выполнение morphing между source и target изображениями
        
        Args:
            source_image: путь к source изображению или PIL Image
            target_image: путь к target изображению или PIL Image
            num_steps: число шагов интерполяции
            reduce_tokens: уменьшать ли число токенов через кластеризацию
            n_clusters: число кластеров для уменьшения токенов
            preprocess_image: применять ли предобработку изображений (удаление фона, обрезка)
            
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
        
        # Предобработка изображений (как в TRELLIS pipeline)
        if preprocess_image:
            print("Preprocessing source image...")
            source_img = self.decoder.pipeline.preprocess_image(source_img)
            print("Preprocessing target image...")
            target_img = self.decoder.pipeline.preprocess_image(target_img)
        
        # Шаг 1: Кодирование изображений через TRELLIS encode_image
        print("Encoding source image...")
        lat_src = self.decoder.pipeline.encode_image([source_img])
        # Преобразуем из [batch, num_patches, dim] в [num_patches, dim]
        lat_src = lat_src.squeeze(0).cpu().numpy()
        
        print("Encoding target image...")
        lat_tgt = self.decoder.pipeline.encode_image([target_img])
        # Преобразуем из [batch, num_patches, dim] в [num_patches, dim]
        lat_tgt = lat_tgt.squeeze(0).cpu().numpy()
        
        # Шаг 2: Вычисление morphing latents через barycenter optimization
        print(f"Computing barycenter sequence ({num_steps} steps)...")
        morphing_latents = self.barycenter_opt.compute_morphing_sequence(
            lat_src,
            lat_tgt,
            num_steps=num_steps,
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
    
    def morph_three_points(
        self,
        source_image: Union[str, Image.Image],
        target1_image: Union[str, Image.Image],
        target2_image: Union[str, Image.Image],
        num_steps: int = 10,
        preprocess_image: bool = True
    ) -> Tuple[List[dict], List[dict]]:
        """
        Генерация траектории между тремя точками:
        1. Из source в target1
        2. Из source в target2
        
        Args:
            source_image: путь к source изображению или PIL Image (точка 1)
            target1_image: путь к первому target изображению или PIL Image (точка 2)
            target2_image: путь ко второму target изображению или PIL Image (точка 3)
            num_steps: число шагов интерполяции для каждой траектории
            preprocess_image: применять ли предобработку изображений (удаление фона, обрезка)
            
        Returns:
            tuple: (results_1_to_2, results_1_to_3) - результаты морфинга из 1 в 2 и из 1 в 3
        """
        # Загрузка изображений
        if isinstance(source_image, str):
            source_img = Image.open(source_image).convert('RGB')
        else:
            source_img = source_image
        
        if isinstance(target1_image, str):
            target1_img = Image.open(target1_image).convert('RGB')
        else:
            target1_img = target1_image
        
        if isinstance(target2_image, str):
            target2_img = Image.open(target2_image).convert('RGB')
        else:
            target2_img = target2_image
        
        # Предобработка изображений
        if preprocess_image:
            print("Preprocessing images...")
            source_img = self.decoder.pipeline.preprocess_image(source_img)
            target1_img = self.decoder.pipeline.preprocess_image(target1_img)
            target2_img = self.decoder.pipeline.preprocess_image(target2_img)
        
        # Кодирование всех изображений
        print("Encoding images...")
        lat_src = self.decoder.pipeline.encode_image([source_img])
        lat_src = lat_src.squeeze(0).cpu().numpy()
        
        lat_tgt1 = self.decoder.pipeline.encode_image([target1_img])
        lat_tgt1 = lat_tgt1.squeeze(0).cpu().numpy()
        
        lat_tgt2 = self.decoder.pipeline.encode_image([target2_img])
        lat_tgt2 = lat_tgt2.squeeze(0).cpu().numpy()
        
        # Траектория 1: из source в target1
        print(f"\n=== Trajectory 1: Source -> Target1 ({num_steps} steps) ===")
        morphing_latents_1_to_2 = self.barycenter_opt.compute_morphing_sequence(
            lat_src,
            lat_tgt1,
            num_steps=num_steps,
        )
        
        print("Decoding trajectory 1...")
        results_1_to_2 = []
        for i, lat_t in enumerate(morphing_latents_1_to_2):
            print(f"  Decoding step {i+1}/{len(morphing_latents_1_to_2)}...")
            outputs = self.decoder.decode(lat_t)
            results_1_to_2.append(outputs)
        
        # Траектория 2: из source в target2
        print(f"\n=== Trajectory 2: Target1 -> Target2 ({num_steps} steps) ===")
        morphing_latents_2_to_3 = self.barycenter_opt.compute_morphing_sequence(
            lat_tgt1,
            lat_tgt2,
            num_steps=num_steps,
        )
        
        print("Decoding trajectory 2...")
        results_2_to_3 = []
        for i, lat_t in enumerate(morphing_latents_2_to_3):
            print(f"  Decoding step {i+1}/{len(morphing_latents_2_to_3)}...")
            outputs = self.decoder.decode(lat_t)
            results_2_to_3.append(outputs)
        
        print("\nThree-point morphing completed!")
        print(f"  Trajectory 1 (Source->Target1): {len(results_1_to_2)} frames")
        print(f"  Trajectory 2 (Source->Target2): {len(results_2_to_3)} frames")
        
        return results_1_to_2 + results_2_to_3
