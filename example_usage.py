"""
Пример использования pipeline для 3D morphing
"""
import os
import numpy as np
import imageio
from morphing_pipeline import MorphingPipeline

# Настройка окружения для TRELLIS (как в example.py)
os.environ['SPCONV_ALGO'] = 'native'


def main():
    """Пример использования morphing pipeline"""
    
    # Инициализация pipeline
    print("Initializing morphing pipeline...")
    # Попытка использовать реальный Trellis decoder
    pipeline = MorphingPipeline(
        dino_model="dinov2_vitl14",
        barycenter_reg=0.1,
        device="cuda",
    )
    
    # Пути к изображениям (замените на ваши пути)
    source_image_path = "source.jpg"
    target_image_path = "target.jpg"
    
    # Проверка существования файлов
    if not os.path.exists(source_image_path):
        print(f"Warning: Source image not found: {source_image_path}")
        print("Please provide valid image paths.")
        return
    
    if not os.path.exists(target_image_path):
        print(f"Warning: Target image not found: {target_image_path}")
        print("Please provide valid image paths.")
        return
    
    # Выполнение morphing
    print(f"\nStarting morphing from {source_image_path} to {target_image_path}...")
    results = pipeline.morph(
        source_image=source_image_path,
        target_image=target_image_path,
        num_steps=4,
        reduce_tokens=False,
        n_clusters=256
    )
    
    print(f"\nMorphing completed! Generated {len(results)} intermediate frames.")
    
    # Создаем директорию для сохранения результатов
    output_dir = "morphing_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to {output_dir}/...")
    from trellis.utils import render_utils, postprocessing_utils
    
    # Сохранение результатов
    for i, outputs in enumerate(results):
        if outputs is None:
            print(f"  Step {i}: Skipping (no outputs)")
            continue
            
        print(f"  Step {i}: Saving results...")
        
        # Создаем GLB файл
        if 'mesh' in outputs and len(outputs['mesh']) > 0 and 'gaussian' in outputs and len(outputs['gaussian']) > 0:
            try:
                glb_path = os.path.join(output_dir, f"mesh_step_{i:03d}.glb")
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                glb.export(glb_path)
                print(f"    Saved GLB: {glb_path}")
            except Exception as e:
                print(f"    Warning: Could not save GLB: {e}")
        
        # Создаем видео из mesh
        if 'mesh' in outputs and len(outputs['mesh']) > 0:
            try:
                video_path = os.path.join(output_dir, f"mesh_step_{i:03d}.mp4")
                video = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
                imageio.mimsave(video_path, video, fps=30)
                print(f"    Saved video: {video_path}")
            except Exception as e:
                print(f"    Warning: Could not save video: {e}")
        
        # Создаем видео из gaussian если доступен
        if 'gaussian' in outputs and len(outputs['gaussian']) > 0:
            try:
                video_path = os.path.join(output_dir, f"gaussian_step_{i:03d}.mp4")
                video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
                imageio.mimsave(video_path, video, fps=30)
                print(f"    Saved gaussian video: {video_path}")
            except Exception as e:
                print(f"    Warning: Could not save gaussian video: {e}")
    
    print(f"\nAll results saved to {output_dir}/")



if __name__ == "__main__":
    main()