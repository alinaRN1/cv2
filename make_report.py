"""Сформировать отчёт по сохранённым результатам лабораторной работы."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    carla = read('results/carla_metrics.json')
    people = read('results/pennfudan_metrics.json')
    carla_run = read('results/carla_run.json')
    people_run = read('results/pennfudan_run.json')
    comparison = read('results/experiment_32/carla_comparison.json')
    external = read('results/experiment_32/carla_external_metrics.json')
    report = f'''# Лабораторная работа № 2
## Сегментация изображений с помощью U-Net

### Цель

Реализовать U-Net и исследовать сегментацию на учебном наборе CARLA. Повторить подготовку данных, обучение и проверку модели на другом наборе — Penn-Fudan для выделения людей.

### Данные

- **CARLA:** 1060 изображений и масок для 23 классов. Разбиение: 848 для обучения, 106 для валидации и 106 для тестирования. [Источник изображений и масок](https://github.com/Rhythm-Adhikari1/U-Net/tree/072afa362ab567ceb8c151b18b00e05ac632bf26/data). Близкие номера кадров объединены в группы.
- **Penn-Fudan:** 170 фотографий людей с размеченными масками. В исходной маске каждому человеку соответствует свой номер; для семантической сегментации все ненулевые номера объединены в класс «человек», 0 — фон. Случайное разбиение с seed=42: 136 изображений для обучения, 17 для валидации и 17 для теста. [Источник набора](https://github.com/swallan/PennFudanPed/tree/ec1d4583fb436b14e2062587c8b28a5018668a5e). Авторы набора: Liming Wang, Jianbo Shi, Gang Song, I-fan Shen, *Object Detection Combining Recognition and Segmentation*, ACCV 2007.

Изображения и маски хранятся по отдельным папкам `images/train`, `images/val`, `images/test` и `masks/train`, `masks/val`, `masks/test`. OpenCV изменяет размер изображения до 96×128, переводит его в RGB и нормализует пиксели. Маски изменяют размер методом nearest neighbor, чтобы сохранить номера классов.

### Модель

Использована U-Net: пять блоков encoder, четыре блока decoder и skip connections. Функция потерь — sparse categorical crossentropy по logits, оптимизатор — Adam, batch size — 8. Базовые модели обучались с 8 начальными фильтрами. Для CARLA также сохранён дополнительный эксперимент с 32 начальными фильтрами.

### CARLA

Обучение выполнено в течение {carla_run['epochs']} эпох; выбрана эпоха {carla_run['best_epoch']} по validation loss. На {carla['test_images']} тестовых изображениях accuracy составила **{carla['accuracy']:.3f}**, mean IoU — **{carla['mean_iou']:.3f}**. Baseline, всегда предсказывающий самый частый обучающий класс, дал mean IoU {carla['majority_baseline']['mean_iou']:.3f}. Крупные области дороги и неба выделяются лучше редких небольших объектов.

![Баланс классов CARLA](results/carla_classes.png)

![Обучение CARLA](results/carla_history.png)

![Предсказания CARLA](results/carla_predictions.png)

![Трудные примеры CARLA](results/carla_worst_predictions.png)

### Penn-Fudan: выделение людей

Для набора людей модель обучалась {people_run['epochs']} эпох. По минимальному validation loss выбрана эпоха {people_run['best_epoch']}. На {people['test_images']} тестовых изображениях получены:

| Метрика | Значение |
| --- | ---: |
| Pixel accuracy | {people['accuracy']:.3f} |
| Mean IoU по двум классам | {people['mean_iou']:.3f} |
| IoU фона | {people['iou_per_class'][0]:.3f} |
| IoU человека | {people['iou_per_class'][1]:.3f} |
| Mean IoU baseline «всегда фон» | {people['majority_baseline']['mean_iou']:.3f} |

![Баланс классов Penn-Fudan](results/pennfudan_classes.png)

![Обучение Penn-Fudan](results/pennfudan_history.png)

![Изображение, истинная маска и предсказание](results/pennfudan_predictions.png)

Модель выделяет заметные фигуры, но может пропустить части человека или объединить соседние фигуры. Тест содержит 17 изображений, поэтому результат даёт лишь приблизительную оценку. Метрики CARLA и Penn-Fudan нельзя напрямую сравнивать: наборы и классы различаются.

### Дополнительный эксперимент CARLA

В модели с 32 начальными фильтрами использована взвешенная функция потерь; выбор эпохи проводился по validation IoU. На тестовой части CARLA получены accuracy {comparison['test_accuracy']['32']:.3f} и mean IoU {comparison['test_mean_iou']['32']:.3f}. Одновременно менялись ширина модели и функция потерь, поэтому их отдельный вклад не определён. Поскольку тест модели с 8 фильтрами уже анализировался, сравнение имеет исследовательский характер.

![Предсказания модели CARLA с 32 фильтрами](results/experiment_32/carla_predictions.png)

![Трудные примеры CARLA с 32 фильтрами](results/experiment_32/carla_worst_predictions.png)

Дополнительно использованы 223 кадра из [другого набора CARLA](https://github.com/cpow-89/carla_image_segmentation_dataset/tree/aa0eb00150f3fda6f7950f097baa145890717979). В нём оцениваются классы 1–12, а неразмеченные пиксели исключены. Mean IoU составил {external['models']['8']['mean_iou']:.3f} для модели с 8 фильтрами и {external['models']['32']['mean_iou']:.3f} для модели с 32 фильтрами. Эти значения относятся к другой схеме оценки.

![Проверка на другом наборе CARLA](results/experiment_32/carla_external_predictions.png)

### Вывод

U-Net обучена на двух задачах: многоклассовой сегментации CARLA и выделении людей из фона. Результаты на тестовых масках подтверждают работу модели, но также показывают ошибки на небольших объектах и границах.

### Собственные фотографии

Обученные модели дополнительно применены к четырём собственным фотографиям. Для этих изображений эталонных масок нет, поэтому IoU не рассчитывался. Модель людей ошибочно выделяет части фона на некоторых снимках; это видно на предсказанных масках.

![Собственные фотографии и модели CARLA](results/carla_own_photos.png)

![Собственные фотографии и выделение людей](results/pennfudan_own_photos.png)
'''
    Path('REPORT.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
