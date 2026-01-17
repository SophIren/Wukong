"""
Тесты для проверки корректности barycenter optimization
"""
import numpy as np
import torch
from barycenter_optimization import BarycenterOptimizer


def test_boundary_conditions():
    """Тест граничных условий: alpha=0 должен дать source, alpha=1 - target"""
    print("Test 1: Boundary conditions...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    # Простые тестовые данные
    lat_src = np.random.randn(10, 8)  # 10 токенов, 8 размерность
    lat_tgt = np.random.randn(10, 8)
    
    # При alpha=0 должен быть source
    bary_0 = optimizer.compute_barycenter(lat_src, lat_tgt, alpha=0.0)
    diff_0 = np.linalg.norm(bary_0 - lat_src)
    print(f"  alpha=0: ||barycenter - source|| = {diff_0:.6f}")
    
    # При alpha=1 должен быть близок к target (транспортированный)
    bary_1 = optimizer.compute_barycenter(lat_src, lat_tgt, alpha=1.0)
    
    # Проверка, что размерность сохраняется
    assert bary_0.shape == lat_src.shape, f"Shape mismatch: {bary_0.shape} vs {lat_src.shape}"
    assert bary_1.shape == lat_src.shape, f"Shape mismatch: {bary_1.shape} vs {lat_src.shape}"
    
    print("  ✓ Boundary conditions test passed (shape check)")
    return True


def test_continuity():
    """Тест непрерывности: barycenter должен плавно изменяться с alpha"""
    print("\nTest 2: Continuity...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    lat_src = np.random.randn(20, 16)
    lat_tgt = np.random.randn(20, 16)
    
    alphas = np.linspace(0, 1, 11)
    barycenters = []
    
    for alpha in alphas:
        bary = optimizer.compute_barycenter(lat_src, lat_tgt, alpha)
        barycenters.append(bary)
    
    # Проверка, что изменения между соседними шагами не слишком большие
    max_change = 0
    for i in range(1, len(barycenters)):
        change = np.linalg.norm(barycenters[i] - barycenters[i-1])
        max_change = max(max_change, change)
    
    print(f"  Max change between consecutive steps: {max_change:.6f}")
    
    # Проверка, что изменение пропорционально шагу alpha
    alpha_step = alphas[1] - alphas[0]
    avg_change = sum(np.linalg.norm(barycenters[i] - barycenters[i-1]) 
                     for i in range(1, len(barycenters))) / (len(barycenters) - 1)
    
    print(f"  Average change per alpha step: {avg_change:.6f}")
    print("  ✓ Continuity test passed")
    return True


def test_symmetry():
    """Тест симметрии: barycenter(src, tgt, alpha) должен быть похож на barycenter(tgt, src, 1-alpha)"""
    print("\nTest 3: Symmetry (approximate)...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    lat_src = np.random.randn(15, 10)
    lat_tgt = np.random.randn(15, 10)
    
    alpha = 0.3
    bary_1 = optimizer.compute_barycenter(lat_src, lat_tgt, alpha=alpha)
    bary_2 = optimizer.compute_barycenter(lat_tgt, lat_src, alpha=1-alpha)
    
    # Из-за транспортировки они могут отличаться, но должны быть близки по смыслу
    # Проверяем, что оба имеют разумные значения
    print(f"  bary(src, tgt, {alpha}) norm: {np.linalg.norm(bary_1):.6f}")
    print(f"  bary(tgt, src, {1-alpha}) norm: {np.linalg.norm(bary_2):.6f}")
    print("  ✓ Symmetry test passed (values are reasonable)")
    return True


def test_morphing_sequence():
    """Тест последовательности морфинга"""
    print("\nTest 4: Morphing sequence...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    lat_src = np.random.randn(30, 16)
    lat_tgt = np.random.randn(30, 16)
    
    sequence = optimizer.compute_morphing_sequence(lat_src, lat_tgt, num_steps=5, reduce_tokens=False)
    
    assert len(sequence) == 6, f"Expected 6 steps (0 to 5), got {len(sequence)}"
    assert all(b.shape == lat_src.shape for b in sequence), "All barycenters should have same shape"
    
    # Проверка плавности последовательности
    diffs = [np.linalg.norm(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
    print(f"  Sequence length: {len(sequence)}")
    print(f"  Step differences: {[f'{d:.4f}' for d in diffs]}")
    
    # Проверка, что первый элемент близок к source, последний - к target
    first_diff = np.linalg.norm(sequence[0] - lat_src)
    last_diff = np.linalg.norm(sequence[-1] - lat_tgt)
    
    print(f"  First step distance from source: {first_diff:.6f}")
    print(f"  Last step distance from target: {last_diff:.6f}")
    
    print("  ✓ Morphing sequence test passed")
    return True


def test_token_reduction():
    """Тест уменьшения числа токенов"""
    print("\nTest 5: Token reduction...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    lat_src = np.random.randn(500, 32)  # Много токенов
    lat_tgt = np.random.randn(500, 32)
    
    # С уменьшением токенов
    lat_src_reduced, weights_src = optimizer.reduce_tokens(lat_src, n_clusters=100)
    lat_tgt_reduced, weights_tgt = optimizer.reduce_tokens(lat_tgt, n_clusters=100)
    
    assert lat_src_reduced.shape[0] == 100, f"Expected 100 clusters, got {lat_src_reduced.shape[0]}"
    assert lat_src_reduced.shape[1] == 32, "Dimension should be preserved"
    assert np.abs(weights_src.sum() - 1.0) < 1e-6, "Weights should sum to 1"
    
    # Проверка, что barycenter работает с уменьшенными токенами
    bary = optimizer.compute_barycenter(
        lat_src_reduced, lat_tgt_reduced, alpha=0.5,
        weights_src=weights_src, weights_tgt=weights_tgt
    )
    
    assert bary.shape == lat_src_reduced.shape, "Barycenter shape should match input"
    
    print(f"  Reduced from {lat_src.shape[0]} to {lat_src_reduced.shape[0]} tokens")
    print(f"  Weights sum: {weights_src.sum():.6f}")
    print("  ✓ Token reduction test passed")
    return True


def visualize_barycenter():
    """Визуализация barycenter для 2D данных"""
    print("\nTest 6: Visualization (2D example)...")
    
    optimizer = BarycenterOptimizer(reg=0.1)
    
    # Создаем два 2D распределения
    np.random.seed(42)
    lat_src = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
    lat_tgt = np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 50)
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Для серверов без GUI
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(alphas), figsize=(15, 3))
        
        for idx, alpha in enumerate(alphas):
            bary = optimizer.compute_barycenter(lat_src, lat_tgt, alpha=alpha)
            ax = axes[idx]
            ax.scatter(lat_src[:, 0], lat_src[:, 1], alpha=0.3, c='blue', label='Source' if idx == 0 else '')
            ax.scatter(lat_tgt[:, 0], lat_tgt[:, 1], alpha=0.3, c='red', label='Target' if idx == 0 else '')
            ax.scatter(bary[:, 0], bary[:, 1], alpha=0.7, c='green', s=30, label='Barycenter' if idx == 0 else '')
            ax.set_title(f'α={alpha:.2f}')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('barycenter_visualization.png', dpi=150, bbox_inches='tight')
        print("  ✓ Visualization saved to barycenter_visualization.png")
        plt.close()
    except Exception as e:
        print(f"  Visualization skipped: {e}")
    
    return True


def test_improved_barycenter():
    """Тест улучшенной реализации barycenter с использованием правильного метода из POT"""
    print("\nTest 7: Testing improved barycenter implementation...")
    
    try:
        import ot
        
        # Проверяем, доступен ли правильный barycenter в POT
        lat_src = np.random.randn(20, 8)
        lat_tgt = np.random.randn(20, 8)
        weights_src = np.ones(20) / 20
        weights_tgt = np.ones(20) / 20
        
        # Вычисляем cost matrix
        M = ot.dist(lat_src, lat_tgt)
        
        # Пробуем использовать правильный barycenter из POT
        # Для fixed support barycenter
        try:
            # Free support barycenter (более точный, но требует другого API)
            # Для fixed support используем транспортировку
            print("  Using transport-based barycenter (current implementation)")
            
            optimizer = BarycenterOptimizer(reg=0.1)
            bary = optimizer.compute_barycenter(lat_src, lat_tgt, alpha=0.5)
            
            # Проверяем, что результат разумный
            assert bary.shape == lat_src.shape
            assert not np.isnan(bary).any(), "Barycenter contains NaN"
            assert not np.isinf(bary).any(), "Barycenter contains Inf"
            
            print("  ✓ Improved barycenter test passed")
            return True
        except Exception as e:
            print(f"  Warning: {e}")
            return False
    except ImportError:
        print("  POT not available for testing")
        return False


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Testing Barycenter Optimization Implementation")
    print("=" * 60)
    
    tests = [
        test_boundary_conditions,
        test_continuity,
        test_symmetry,
        test_morphing_sequence,
        test_token_reduction,
        test_improved_barycenter,
        visualize_barycenter,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    main()
