"""
Пример использования Trellis2-based morphing pipeline.

Аналогичен `example_usage.py`, но:
- использует Trellis2Encoder (через Trellis2ImageTo3DPipeline.get_cond)
- декодирует латенты через `Trellis2Decoder`.
"""
import os
import numpy as np
import imageio
from PIL import Image

from barycenter_optimization import BarycenterOptimizer
from trellis2_decoder import Trellis2Decoder


# Базовые настройки окружения (по аналогии с TRELLIS)
os.environ["SPCONV_ALGO"] = "native"


def encode_image_trellis2(decoder: Trellis2Decoder, img: Image.Image, resolution: int = 512) -> np.ndarray:
    """
    Получить латенты из Trellis2 (аналог encode_image в TRELLIS v1).

    Возвращает numpy-массив формы [num_tokens, dim].
    """
    if img.mode != "RGBA":
        img = img.convert("RGB")

    # Используем внутренний image_cond_model через get_cond
    cond = decoder.pipeline.get_cond([img], resolution=resolution, include_neg_cond=False)
    feats = cond["cond"]  # [1, num_tokens, dim] или подобная форма
    feats = feats.squeeze(0).cpu().numpy()
    return feats


def main():
    """Пример использования Trellis2 morphing pipeline."""

    print("Initializing Trellis2 morphing components...")
    barycenter_opt = BarycenterOptimizer()
    decoder = Trellis2Decoder(device="cuda")

    # Пути к изображениям (замените на ваши пути)
    source_image_path = "shape_000.jpg"
    target_image_path = "shape_002.jpg"

    # Проверка существования файлов
    if not os.path.exists(source_image_path):
        print(f"Warning: Source image not found: {source_image_path}")
        print("Please provide valid image paths.")
        return

    if not os.path.exists(target_image_path):
        print(f"Warning: Target image not found: {target_image_path}")
        print("Please provide valid image paths.")
        return

    # Загрузка изображений
    source_img = Image.open(source_image_path).convert("RGB")
    target_img = Image.open(target_image_path).convert("RGB")

    # Шаг 1: кодирование изображений в Trellis2 латенты
    print("Encoding images with Trellis2...")
    lat_src = encode_image_trellis2(decoder, source_img, resolution=512)
    lat_tgt = encode_image_trellis2(decoder, target_img, resolution=512)

    # Шаг 2: вычисление последовательности морфинг-латентов
    num_steps = 4
    print(f"Computing barycenter sequence ({num_steps} steps)...")
    morphing_latents = barycenter_opt.compute_morphing_sequence(
        lat_src,
        lat_tgt,
        num_steps=num_steps,
    )

    # Шаг 3: декодирование каждого латента через Trellis2Decoder
    print("Decoding latents to 3D meshes with Trellis2...")
    all_meshes = []
    for i, lat in enumerate(morphing_latents):
        print(f"  Decoding step {i + 1}/{len(morphing_latents)}...")
        meshes = decoder.decode(lat, pipeline_type="1024_cascade")
        all_meshes.append(meshes)

    print(f"\nMorphing completed! Generated {len(all_meshes)} intermediate frames.")

    # Создаем директорию для сохранения результатов
    output_dir = "morphing_output_trellis2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to {output_dir}/...")

    # Импорты для рендера/экспорта
    from trellis2.utils import render_utils
    import o_voxel

    for i, meshes in enumerate(all_meshes):
        if not meshes:
            print(f"  Step {i}: Skipping (no meshes)")
            continue

        mesh = meshes[0]
        print(f"  Step {i}: Saving results...")

        # GLB
        try:
            glb_path = os.path.join(output_dir, f"trellis2_step_{i:03d}.glb")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=4096,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(glb_path, extension_webp=True)
            print(f"    Saved GLB: {glb_path}")
        except Exception as e:
            print(f"    Warning: Could not save GLB: {e}")

        # Видео
        try:
            video_path = os.path.join(output_dir, f"trellis2_step_{i:03d}.mp4")
            video_frames = render_utils.make_pbr_vis_frames(
                render_utils.render_video(mesh)
            )
            imageio.mimsave(video_path, video_frames, fps=15)
            print(f"    Saved video: {video_path}")
        except Exception as e:
            print(f"    Warning: Could not save video: {e}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()

