# Wukong's 72 Transformations: High-fidelity Textured 3D Morphing

Реализация метода из статьи "Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models" для создания textured 3D morphing между изображениями.

## Архитектура

Pipeline состоит из трех основных компонентов:

1. **DINOv2 Encoder** - кодирование source и target изображений в латентные представления
2. **Barycenter Optimization** - решение задачи barycenter optimization для получения промежуточных морфинг латентов
3. **Trellis1 Decoder** - декодирование латентов в textured 3D mesh

## Установка

### Базовые зависимости

```bash
pip install -r requirements.txt
```

### Установка TRELLIS для 3D декодирования

Для работы с реальным Trellis1 декодером необходимо установить TRELLIS. Есть несколько вариантов:

**Вариант 1: Клонирование репозитория TRELLIS (рекомендуется)**

```bash
git clone https://github.com/microsoft/TRELLIS
cd TRELLIS
pip install -e .
```

Затем установите переменную окружения для пути к TRELLIS:
```bash
export TRELLIS_PATH=/path/to/TRELLIS
```

**Вариант 2: Использование pip (если доступно)**

```bash
pip install trellis-3d
```

**Вариант 3: Установка только необходимых зависимостей**

Если TRELLIS не установлен, код будет использовать placeholder decoder. Для работы с mesh рекомендуется установить:
```bash
pip install trimesh open3d
```

### Проверка установки

После установки проверьте, что TRELLIS доступен:

```python
from trellis_decoder import TrellisDecoder
decoder = TrellisDecoder(device="cuda")  # Если TRELLIS установлен, модель загрузится
```

## Использование

### Базовый пример

```python
from morphing_pipeline import MorphingPipeline

# Инициализация pipeline
pipeline = MorphingPipeline(
    dino_model="dinov2_vitb14",
    barycenter_reg=0.1,
    device="cuda"
)

# Выполнение morphing
results = pipeline.morph(
    source_image="source.jpg",
    target_image="target.jpg",
    num_steps=10,
    reduce_tokens=True,
    n_clusters=256
)

# results содержит список кортежей (mesh, texture) для каждого шага
for i, (mesh, texture) in enumerate(results):
    # Сохранение результатов
    save_mesh(mesh, f"output_step_{i}.obj")
```

### Один шаг morphing

```python
# Получение промежуточного состояния для alpha=0.5
mesh, texture = pipeline.morph_step(
    source_image="source.jpg",
    target_image="target.jpg",
    alpha=0.5
)
```

## Структура проекта

```
wukong/
├── encoder_dinov2.py          # DINOv2 encoder для кодирования изображений
├── barycenter_optimization.py # Barycenter optimization для morphing latents
├── trellis_decoder.py         # Trellis1 decoder для textured 3D morphing
├── morphing_pipeline.py       # Основной pipeline
├── example_usage.py           # Пример использования
├── requirements.txt           # Зависимости
└── README.md                  # Документация
```

## Компоненты

### DINOv2 Encoder (`encoder_dinov2.py`)

- Загружает предобученную модель DINOv2
- Кодирует изображения в латентные представления (patch tokens)
- Поддерживает различные размеры моделей DINOv2 (ViT-S/B/L/G)

### Barycenter Optimization (`barycenter_optimization.py`)

- Решает задачу Wasserstein barycenter optimization между латентами
- Использует библиотеку POT (Python Optimal Transport)
- Поддерживает энтропийную регуляризацию для гладкости
- Опциональное уменьшение числа токенов через кластеризацию
- Рекурсивная инициализация для последовательных шагов

### Trellis Decoder (`trellis_decoder.py`)

- Интерфейс для интеграции с TRELLIS-image-to-3D
- Декодирует латенты в textured 3D mesh
- Поддерживает placeholder decoder для тестирования без Trellis

## Параметры

### MorphingPipeline

- `dino_model`: модель DINOv2 ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")
- `barycenter_reg`: параметр энтропийной регуляризации (больше = более гладкий результат)
- `device`: устройство для вычислений ("cuda" или "cpu")

### morph()

- `num_steps`: число промежуточных шагов интерполяции
- `reduce_tokens`: уменьшать ли число токенов через K-means кластеризацию
- `n_clusters`: число кластеров при уменьшении токенов

## Примечания

- Для texture-controlled 3D morphing необходимы дополнительные модификации (не включены в данную реализацию)
- Интеграция с TRELLIS требует установки соответствующей библиотеки
- При использовании placeholder decoder реальное 3D не генерируется

## Лицензия

См. лицензию проекта.