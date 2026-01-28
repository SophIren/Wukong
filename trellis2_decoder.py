"""
Trellis2 Decoder для image-to-3D с латентами.

По смыслу аналогичен `TrellisDecoder`, но использует Trellis2 pipeline.
"""
from typing import List, Union, Optional, Dict

import numpy as np
import torch


class Trellis2Decoder:
    """
    Trellis2 decoder, который принимает латенты (из морфинга)
    и декодирует их в 3D через `Trellis2ImageTo3DPipeline`.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "microsoft/TRELLIS.2-4B",
    ):
        """
        Инициализация Trellis2 decoder.

        Args:
            device: устройство для вычислений ("cuda" или "cpu").
            model_name: имя/путь модели Trellis2, передаётся в
                `Trellis2ImageTo3DPipeline.from_pretrained`.
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        self.pipeline = None

        # Загрузка Trellis2 модели
        self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка предобученной Trellis2 модели.
        """
        import os
        import sys

        # Добавляем путь к репозиторию TRELLIS.2, если он не в PYTHONPATH
        trellis2_path = os.environ.get("TRELLIS2_PATH", "./TRELLIS.2")
        if os.path.exists(trellis2_path) and trellis2_path not in sys.path:
            sys.path.insert(0, trellis2_path)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        print(f"Loading Trellis2 model: {self.model_name}...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_name)

        # Перемещаем на нужное устройство
        if self.device == "cuda":
            self.pipeline.to(torch.device("cuda"))
        else:
            self.pipeline.to(torch.device("cpu"))

        print("Trellis2 pipeline loaded successfully!")

    def decode(
        self,
        latents: Union[np.ndarray, torch.Tensor],
        pipeline_type: str = "1024_cascade",
        sparse_structure_sampler_params: Optional[Dict] = None,
        shape_slat_sampler_params: Optional[Dict] = None,
        tex_slat_sampler_params: Optional[Dict] = None,
        max_num_tokens: int = 49152,
    ) -> List:
        """
        Декодирование латентов (аналогично `TrellisDecoder.decode`).

        Args:
            latents: морфинг-латенты [num_tokens, dim] или [dim].
            pipeline_type: тип пайплайна ('512', '1024', '1024_cascade', '1536_cascade').
            *_sampler_params: дополнительные параметры для семплеров.
            max_num_tokens: максимум токенов для cascade-пайплайнов.

        Returns:
            Список `MeshWithVoxel`, как у `Trellis2ImageTo3DPipeline.run`.
        """
        # Приведение к torch тензору
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()

        # Нормализация формы латентов:
        # [dim] -> [1, 1, dim], [num_tokens, dim] -> [1, num_tokens, dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0).unsqueeze(0)
        elif latents.dim() == 2:
            latents = latents.unsqueeze(0)
        elif latents.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected latents shape: {latents.shape}")

        latents = latents.to(self.device)

        with torch.no_grad():
            meshes = self._decode_with_pipeline(
                latents=latents,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params=sparse_structure_sampler_params or {},
                shape_slat_sampler_params=shape_slat_sampler_params or {},
                tex_slat_sampler_params=tex_slat_sampler_params or {},
                max_num_tokens=max_num_tokens,
            )

        return meshes

    def _decode_with_pipeline(
        self,
        latents: torch.Tensor,
        pipeline_type: str,
        sparse_structure_sampler_params: Dict,
        shape_slat_sampler_params: Dict,
        tex_slat_sampler_params: Dict,
        max_num_tokens: int,
    ) -> List:
        """
        Логика декодирования, повторяющая структуру `Trellis2ImageTo3DPipeline.run`,
        но вместо `get_cond(image)` мы используем готовые латенты как `cond`.
        """
        # Создаём cond аналогично тому, что возвращает get_cond
        cond_512 = {
            "cond": latents,
            "neg_cond": torch.zeros_like(latents),
        }
        cond_1024 = None
        if pipeline_type != "512":
            cond_1024 = {
                "cond": latents,
                "neg_cond": torch.zeros_like(latents),
            }

        # Выбор разрешения sparse structure, как в run()
        ss_res = {
            "512": 32,
            "1024": 64,
            "1024_cascade": 32,
            "1536_cascade": 32,
        }[pipeline_type]

        # 1) Sparse structure
        coords = self.pipeline.sample_sparse_structure(
            cond=cond_512,
            resolution=ss_res,
            num_samples=1,
            sampler_params=sparse_structure_sampler_params,
        )

        # 2) Shape / texture SLAT + 3) Decode latent
        if pipeline_type == "512":
            shape_slat = self.pipeline.sample_shape_slat(
                cond=cond_512,
                flow_model=self.pipeline.models["shape_slat_flow_model_512"],
                coords=coords,
                sampler_params=shape_slat_sampler_params,
            )
            tex_slat = self.pipeline.sample_tex_slat(
                cond=cond_512,
                flow_model=self.pipeline.models["tex_slat_flow_model_512"],
                shape_slat=shape_slat,
                sampler_params=tex_slat_sampler_params,
            )
            res = 512
        elif pipeline_type == "1024":
            shape_slat = self.pipeline.sample_shape_slat(
                cond=cond_1024,
                flow_model=self.pipeline.models["shape_slat_flow_model_1024"],
                coords=coords,
                sampler_params=shape_slat_sampler_params,
            )
            tex_slat = self.pipeline.sample_tex_slat(
                cond=cond_1024,
                flow_model=self.pipeline.models["tex_slat_flow_model_1024"],
                shape_slat=shape_slat,
                sampler_params=tex_slat_sampler_params,
            )
            res = 1024
        elif pipeline_type == "1024_cascade":
            shape_slat, res = self.pipeline.sample_shape_slat_cascade(
                lr_cond=cond_512,
                cond=cond_1024,
                flow_model_lr=self.pipeline.models["shape_slat_flow_model_512"],
                flow_model=self.pipeline.models["shape_slat_flow_model_1024"],
                lr_resolution=512,
                resolution=1024,
                coords=coords,
                sampler_params=shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
            )
            tex_slat = self.pipeline.sample_tex_slat(
                cond=cond_1024,
                flow_model=self.pipeline.models["tex_slat_flow_model_1024"],
                shape_slat=shape_slat,
                sampler_params=tex_slat_sampler_params,
            )
        elif pipeline_type == "1536_cascade":
            shape_slat, res = self.pipeline.sample_shape_slat_cascade(
                lr_cond=cond_512,
                cond=cond_1024,
                flow_model_lr=self.pipeline.models["shape_slat_flow_model_512"],
                flow_model=self.pipeline.models["shape_slat_flow_model_1024"],
                lr_resolution=512,
                resolution=1536,
                coords=coords,
                sampler_params=shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
            )
            tex_slat = self.pipeline.sample_tex_slat(
                cond=cond_1024,
                flow_model=self.pipeline.models["tex_slat_flow_model_1024"],
                shape_slat=shape_slat,
                sampler_params=tex_slat_sampler_params,
            )
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        torch.cuda.empty_cache()
        meshes = self.pipeline.decode_latent(shape_slat, tex_slat, res)
        return meshes
