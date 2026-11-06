"""Создание отчёта по сохранённым результатам."""
import csv
import json
from pathlib import Path


def scores(name):
    return json.loads(Path(f"results/{name}_metrics.json").read_text(encoding="utf-8"))


def best_epoch(path, metric, maximize=False):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    best = (max if maximize else min)(rows, key=lambda row: float(row[metric]))
    return int(best["epoch"]) + 1, len(rows)


def main():
    carla = scores("carla_unet")
    people = scores("penn32")
    carla_best, carla_epochs = best_epoch("results/carla_unet_history.csv", "val_mean_iou", True)
    people_best, people_epochs = best_epoch("results/penn32_history.csv", "val_loss")
    text = f'''# Лабораторная работа № 2
## Сегментация изображений с помощью U-Net

### Цель работы

Построить модель U-Net, обучить её для сегментации изображений и проверить работу на двух наборах данных.

### 1. Данные

Для первой части использован [набор CARLA](https://github.com/Rhythm-Adhikari1/U-Net/tree/072afa362ab567ceb8c151b18b00e05ac632bf26/data). Он содержит 1060 изображений дорожных сцен и соответствующих масок с 23 классами. На обучение отведено 848 изображений, на валидацию и тест — по 106. Близкие по номеру кадры объединены в группы перед разбиением.

Для второй части выбран [набор Penn-Fudan](https://github.com/swallan/PennFudanPed/tree/ec1d4583fb436b14e2062587c8b28a5018668a5e), содержащий 170 изображений людей с масками, подготовленными авторами набора. Использованы 136 изображений для обучения, 17 для валидации и 17 для тестирования. Все номера людей в исходных масках объединены в один класс: 0 — фон, 1 — человек. Разбиение выполнено случайно с seed=42.

Изображения и маски организованы в папках `images/train`, `images/val`, `images/test` и `masks/train`, `masks/val`, `masks/test`. Каждому изображению соответствует маска с тем же именем.

### 2. Предобработка

При загрузке в модель OpenCV приводит изображения к размеру 96×128, переводит их в RGB и нормализует в диапазон [0, 1]. Маски изменяются до такого же размера методом nearest neighbor, чтобы сохранить номера классов. Для масок CARLA номер класса считывается из красного канала.

### 3. Модель U-Net

Сеть содержит пять блоков сжатия и четыре блока восстановления. Между соответствующими уровнями используются skip connections. Начальное число фильтров — 32; в двух последних блоках сжатия используется Dropout 0,3. На выходе модель имеет 23 канала для CARLA и 2 канала для Penn-Fudan.

Использованы оптимизатор Adam и sparse categorical crossentropy по logits. Обучение модели CARLA проводилось {carla_epochs} эпох, модели Penn-Fudan — {people_epochs} эпох; размер пакета — 8. Для CARLA функция потерь взвешена по частотам классов в обучающих масках; сохранена модель с лучшим validation mean IoU на эпохе {carla_best}. Для Penn-Fudan сохранена модель с минимальным validation loss на эпохе {people_best}.

### 4. Результаты CARLA

На {carla['test_images']} тестовых изображениях получены pixel accuracy **{carla['accuracy']:.3f}** и mean IoU **{carla['mean_iou']:.3f}**. Крупные области, такие как дорога и небо, определяются лучше небольших объектов. На изображениях ниже показаны исходные кадры, настоящие и предсказанные маски.

![Графики обучения CARLA](results/carla_unet_history.png)

![Предсказания CARLA](results/carla_unet_predictions.png)

### 5. Результаты Penn-Fudan

Для оценки использовано {people['test_images']} тестовых изображений. Pixel accuracy составила **{people['accuracy']:.3f}**, mean IoU по двум классам — **{people['mean_iou']:.3f}**, IoU класса «человек» — **{people['iou_per_class'][1]:.3f}**.

![Графики обучения Penn-Fudan](results/penn32_history.png)

![Предсказания Penn-Fudan](results/penn32_predictions.png)

Модель выделяет основные фигуры, но ошибается на границах и может объединять людей, находящихся рядом. Оценка получена на небольшой тестовой выборке.

### 6. Вывод

В работе построена U-Net и выполнена сегментация двух наборов изображений. Для CARLA сеть определяет 23 класса, для Penn-Fudan — человека и фон. На примерах видны как выделенные области, так и ошибки модели.
'''
    Path("REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
