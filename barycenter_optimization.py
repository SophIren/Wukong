"""
Barycenter Optimization для вычисления morphing latents между source и target
"""
import numpy as np
import torch
import ot
from typing import Tuple, Optional
from sklearn.cluster import KMeans


class BarycenterOptimizer:
    """Класс для решения barycenter optimization problem между латентами"""
    
    def __init__(self, reg: float = 0.1, max_iter: int = 1000):
        """
        Инициализация оптимизатора barycenter
        
        Args:
            reg: параметр энтропийной регуляризации (больше = более гладкий результат)
            max_iter: максимальное число итераций для оптимизации (увеличено для лучшей сходимости)
        """
        self.reg = reg
        self.max_iter = max_iter
    
    def compute_cost_matrix(self, lat_src: np.ndarray, lat_tgt: np.ndarray) -> np.ndarray:
        """
        Вычисление cost matrix между source и target латентами
        
        Args:
            lat_src: source латенты [n_src, dim]
            lat_tgt: target латенты [n_tgt, dim]
            
        Returns:
            M: cost matrix [n_src, n_tgt]
        """
        # Евклидово расстояние между латентами
        M = ot.dist(lat_src, lat_tgt, metric='euclidean')
        return M
    
    def reduce_tokens(self, latents: np.ndarray, n_clusters: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Уменьшение числа токенов через кластеризацию (опционально для ускорения)
        
        Args:
            latents: латенты [n_tokens, dim]
            n_clusters: число кластеров (центроидов)
            
        Returns:
            reduced_latents: уменьшенные латенты [n_clusters, dim]
            weights: веса для каждого центроида
        """
        if latents.shape[0] <= n_clusters:
            # Если токенов уже меньше, возвращаем как есть
            weights = np.ones(latents.shape[0]) / latents.shape[0]
            return latents, weights
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latents)
        
        # Центроиды кластеров
        centroids = kmeans.cluster_centers_
        
        # Веса пропорциональны размеру кластеров
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        weights = cluster_sizes.astype(float) / len(labels)
        
        return centroids, weights
    
    def compute_barycenter(
        self, 
        lat_src: np.ndarray, 
        lat_tgt: np.ndarray,
        alpha: float,
        weights_src: Optional[np.ndarray] = None,
        weights_tgt: Optional[np.ndarray] = None,
        init_support: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Вычисление barycenter между source и target латентами
        
        Args:
            lat_src: source латенты [n_src, dim]
            lat_tgt: target латенты [n_tgt, dim]
            alpha: интерполяционный параметр [0, 1] (0 = source, 1 = target)
            weights_src: веса для source распределения (uniform если None)
            weights_tgt: веса для target распределения (uniform если None)
            init_support: начальная поддержка для инициализации (для рекурсивной оптимизации)
            
        Returns:
            barycenter: barycenter латенты
        """
        # Приведение к numpy если torch тензор
        if isinstance(lat_src, torch.Tensor):
            lat_src = lat_src.cpu().numpy()
        if isinstance(lat_tgt, torch.Tensor):
            lat_tgt = lat_tgt.cpu().numpy()
        
        # Унификация размерности
        n_src, dim_src = lat_src.shape
        n_tgt, dim_tgt = lat_tgt.shape
        
        assert dim_src == dim_tgt, f"Размерности латентов не совпадают: {dim_src} != {dim_tgt}"
        
        # Инициализация весов (uniform если не заданы)
        if weights_src is None:
            weights_src = np.ones(n_src) / n_src
        if weights_tgt is None:
            weights_tgt = np.ones(n_tgt) / n_tgt
        
        # Вычисление cost matrix
        M = self.compute_cost_matrix(lat_src, lat_tgt)
        
        # Вычисление Wasserstein barycenter через оптимальный транспорт
        # Для двух распределений с весами (1-alpha, alpha)
        
        if init_support is not None and isinstance(init_support, torch.Tensor):
            init_support = init_support.cpu().numpy()
        
        # Вычисляем транспортировочный план от source к target через Sinkhorn
        # Transport plan T[i,j] - количество массы, транспортированной из source[i] в target[j]
        transport_plan = ot.sinkhorn(weights_src, weights_tgt, M, reg=self.reg, numItermax=self.max_iter, warn=True)
        
        # Транспортируем target токены в пространство source через transport plan
        # Формула: transported_tgt[i] = sum_j (T[i,j] * target[j]) / sum_j T[i,j]
        # Это взвешенное среднее target токенов, транспортированных в позицию i
        transported_tgt = transport_plan @ lat_tgt  # [n_src, dim]
        
        # Нормализация: делим на маргинал по строкам (сумма по target для каждого source)
        transport_marginals = transport_plan.sum(axis=1, keepdims=True)  # [n_src, 1]
        eps = 1e-10
        transported_tgt = transported_tgt / (transport_marginals + eps)
        
        # Интерполяция между source и транспортированными target токенами
        # При alpha=0 получаем source, при alpha=1 - транспортированный target
        barycenter = (1 - alpha) * lat_src + alpha * transported_tgt
        
        # Сохраняем исходный размер выходного barycenter
        assert barycenter.shape == lat_src.shape, "Barycenter должен иметь размер как source"
        
        # Если есть init_support и мы хотим использовать рекурсивную инициализацию,
        # можно добавить взвешенную комбинацию
        if init_support is not None and init_support.shape[0] == barycenter.shape[0]:
            # Смешиваем с предыдущим результатом для плавности
            mix_weight = 0.1  # Небольшой вес для предыдущего результата
            barycenter = (1 - mix_weight) * barycenter + mix_weight * init_support
        
        return barycenter
    
    def compute_morphing_sequence(
        self,
        lat_src: np.ndarray,
        lat_tgt: np.ndarray,
        num_steps: int = 10,
        reduce_tokens: bool = True,
        n_clusters: int = 256
    ) -> list:
        """
        Вычисление последовательности morphing latents
        
        Args:
            lat_src: source латенты
            lat_tgt: target латенты
            num_steps: число шагов интерполяции
            reduce_tokens: уменьшать ли число токенов через кластеризацию
            n_clusters: число кластеров для уменьшения токенов
            
        Returns:
            morphing_latents: список латентов для каждого шага
        """
        # Приведение к numpy если torch тензор
        if isinstance(lat_src, torch.Tensor):
            lat_src = lat_src.cpu().numpy()
        if isinstance(lat_tgt, torch.Tensor):
            lat_tgt = lat_tgt.cpu().numpy()
        
        # Опциональное уменьшение числа токенов
        if reduce_tokens:
            lat_src, weights_src = self.reduce_tokens(lat_src, n_clusters)
            lat_tgt, weights_tgt = self.reduce_tokens(lat_tgt, n_clusters)
        else:
            weights_src = None
            weights_tgt = None
        
        morphing_latents = []
        prev_barycenter = None
        
        # Генерируем последовательность для alpha в [0, 1]
        alphas = np.linspace(0, 1, num_steps + 1)
        
        for alpha in alphas:
            barycenter = self.compute_barycenter(
                lat_src,
                lat_tgt,
                alpha,
                weights_src=weights_src,
                weights_tgt=weights_tgt,
                init_support=prev_barycenter
            )
            
            morphing_latents.append(barycenter)
            prev_barycenter = barycenter  # Рекурсивная инициализация
        
        return morphing_latents