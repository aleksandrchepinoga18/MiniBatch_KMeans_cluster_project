# MiniBatch_KMeans_cluster_project

# Проект: Сжатие изображений с использованием K-means и MiniBatchKMeans

## Описание
Этот проект демонстрирует применение алгоритмов кластеризации для сжатия изображений. Основная цель — уменьшить количество цветов в изображении с помощью методов K-means и MiniBatchKMeans.

## Используемые библиотеки
- `numpy`
- `matplotlib.pyplot`
- `cv2` (OpenCV)
- `sklearn.cluster`

## Методы и алгоритмы
- K-means
- MiniBatchKMeans
- Нормализация данных
- Визуализация

## Краткое описание
1. Загрузка изображения.
2. Нормализация данных.
3. Кластеризация с использованием K-means и MiniBatchKMeans.
4. Визуализация результатов.
5. Анализ и сравнение результатов.

## Итоги
- Успешное сжатие изображения до 16 цветов.
- MiniBatchKMeans показал схожие результаты с K-means, но работает быстрее.
- Наглядная визуализация результатов.

## Примеры кода
```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
img = plt.imread('path_to_image.jpg')

# Преобразуем изображение в массив пикселей
h, w, c = img.shape
pixels = img.reshape(-1, 3)

# Применяем KMeans для уменьшения количества цветов до 16
kmeans = KMeans(n_clusters=16)
kmeans.fit(pixels)
colors = kmeans.cluster_centers_  # Центры кластеров (16 цветов)
labels = kmeans.labels_  # Метки кластеров для каждого пикселя

# Перекрашиваем изображение
img_recolored = colors[labels].reshape(img.shape)

# Визуализация
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(img)
ax[0].set_title('Исходное изображение', size=16)
ax[1].imshow(img_recolored)
ax[1].set_title('Изображение в 16 цветах', size=16)

plt.show()
 
