"""
Trellis1 Decoder для textured 3D morphing
"""
import torch
import torch.nn as nn
import numpy as np


class TrellisDecoder:
    """
    Trellis decoder для декодирования латентов в textured 3D морфинг
    Использует реальный TRELLIS-image-to-3D pipeline
    """
    
    def __init__(self, device: str = "cuda", model_name: str = "microsoft/TRELLIS-image-large"):
        """
        Инициализация Trellis decoder
        
        Args:
            device: устройство для вычислений
            model_name: имя модели TRELLIS ("trellis-image-large", "trellis-image-base")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pipeline = None
        
        # Загрузка Trellis модели
        self._load_model()
    
    def _load_model(self):
        """
        Загрузка предобученной Trellis модели из microsoft/TRELLIS
        """
        import sys
        import os
        
        # Альтернативный импорт (если структура репозитория отличается)
        trellis_path = os.environ.get('TRELLIS_PATH', './TRELLIS')
        if os.path.exists(trellis_path):
            sys.path.insert(0, trellis_path)
        
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        print(f"Loading TRELLIS model: {self.model_name}...")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_name)
        
        # Перемещаем на нужное устройство
        if self.device == "cuda":
            self.pipeline.cuda()
        else:
            self.pipeline.cpu()
        
        self.pipeline.eval()
        print("TRELLIS pipeline loaded successfully!")
    
    def decode(self, latents: np.ndarray) -> dict:
        """
        Декодирование латентов в 3D outputs
        
        Args:
            latents: морфинг латенты [num_tokens, dim] из DINOv2 или [dim]
            
        Returns:
            outputs: словарь с 'mesh', 'gaussian', 'radiance_field'
        """
        # Приведение к torch тензор
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()
        
        # Нормализация формы латентов
        if latents.dim() == 1:
            # Один вектор - расширяем для batch
            latents = latents.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        elif latents.dim() == 2:
            # [num_tokens, dim] - добавляем batch dimension
            latents = latents.unsqueeze(0)  # [1, num_tokens, dim]
        elif latents.dim() == 3:
            # Уже есть batch dimension
            pass
        
        latents = latents.to(self.device)
        
        # Декодирование через Trellis
        with torch.no_grad():
            outputs = self._decode_with_pipeline(latents)
        
        return outputs
    
    def _decode_with_pipeline(self, latents: torch.Tensor) -> dict:
        """
        Декодирование через TrellisImageTo3DPipeline
        Использует готовые DINOv2 эмбеддинги вместо кодирования изображения
        """
        import torch.nn.functional as F
        
        # Нормализуем форму латентов до формата [batch, num_patches, feature_dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0).unsqueeze(0)
        elif latents.dim() == 2:
            latents = latents.unsqueeze(0)
        
        # Применяем layer_norm как в encode_image pipeline
        patchtokens = F.layer_norm(latents, latents.shape[-1:])
        patchtokens = patchtokens.to(self.device)
        
        # Проверяем размерность
        flow_model = self.pipeline.models['sparse_structure_flow_model']
        if patchtokens.shape[-1] != flow_model.cond_channels:
            raise ValueError(
                f"DINOv2 feature dimension ({patchtokens.shape[-1]}) does not match model cond_channels ({flow_model.cond_channels}). "
                f"Please use DINOv2 Large model (dinov2_vitl14 or dinov2_vitl14_reg)."
            )
        
        # Создаем cond напрямую с нашими латентами
        cond = {
            'cond': patchtokens,
            'neg_cond': torch.zeros_like(patchtokens)
        }
        
        # Sample sparse structure
        coords = self.pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params={})
        coords = coords.to(self.device).int()
        
        # Sample structured latent (SLAT)
        slat = self.pipeline.sample_slat(cond, coords, sampler_params={})
        
        # Decode SLAT to all formats
        outputs = self.pipeline.decode_slat(slat, formats=['mesh', 'gaussian', 'radiance_field'])
        
        return outputs