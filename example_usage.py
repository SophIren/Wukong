"""
Пример использования pipeline для 3D morphing
"""
import os
from morphing_pipeline import MorphingPipeline


def main():
    """Пример использования morphing pipeline"""
    
    # Инициализация pipeline
    print("Initializing morphing pipeline...")
    # Попытка использовать реальный Trellis decoder
    # Если TRELLIS не установлен, автоматически используется placeholder
    pipeline = MorphingPipeline(
        dino_model="dinov2_vitb14",
        barycenter_reg=0.1,
        device="cuda",
        use_placeholder_decoder=False  # Попытка загрузить реальный Trellis decoder
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
        num_steps=10,
        reduce_tokens=True,
        n_clusters=256
    )
    
    print(f"\nMorphing completed! Generated {len(results)} intermediate frames.")
    
    # Сохранение результатов (пример)
    # В реальной реализации здесь должна быть логика сохранения mesh и texture
    for i, (mesh, texture) in enumerate(results):
        print(f"  Step {i}: mesh vertices shape: {mesh['vertices'].shape if mesh else 'None'}")
        # Здесь можно сохранить mesh и texture, например, в .obj или .ply формат
        # save_mesh(mesh, f"output_step_{i}.obj")
        # save_texture(texture, f"output_step_{i}.png")
    
    # Пример: получение одного шага
    print("\nComputing single morphing step (alpha=0.5)...")
    mesh, texture = pipeline.morph_step(
        source_image=source_image_path,
        target_image=target_image_path,
        alpha=0.5
    )
    print(f"Single step result: mesh shape: {mesh['vertices'].shape if mesh else 'None'}")


if __name__ == "__main__":
    main()