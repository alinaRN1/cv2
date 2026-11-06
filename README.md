# Лабораторная работа № 2 — сегментация изображений

В работе используются U-Net и два набора:

1. Учебный CARLA — дорожные изображения и маски 23 классов.
2. Penn-Fudan — фотографии и маски людей. Для второго эксперимента все люди объединены в один класс, второй класс — фон.

Основные файлы: `CV_L2.ipynb` — код и выполненные ячейки, `REPORT.md` — отчёт, `lab2.py` — функции U-Net и оценки, `prepare_pennfudan.py` — подготовка наборов. В конце ноутбука и отчёта приведены предсказания для собственных фотографий из папки `photos/`.

Источники: [CARLA](https://github.com/Rhythm-Adhikari1/U-Net/tree/072afa362ab567ceb8c151b18b00e05ac632bf26/data), [Penn-Fudan](https://github.com/swallan/PennFudanPed/tree/ec1d4583fb436b14e2062587c8b28a5018668a5e), [дополнительные кадры CARLA](https://github.com/cpow-89/carla_image_segmentation_dataset/tree/aa0eb00150f3fda6f7950f097baa145890717979). Данные сохраняются в `data/` и в архив работы не включены; авторские условия указаны в источниках.

## Запуск

Нужны Python 3.11 или 3.12, Git и подключение к сети для скачивания данных:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt notebook
python -m ipykernel install --user --name cv-lab2 --display-name "Python (CV Lab 2)"
python prepare_pennfudan.py
jupyter notebook CV_L2.ipynb
```

В Windows вместо команды активации используйте `.venv\Scripts\activate`. Подготовка создаёт папки `images/train`, `images/val`, `images/test`, `masks/train`, `masks/val` и `masks/test` для каждого набора. В `data/pennfudan/` хранится преобразованная бинарная маска: 0 — фон, 1 — человек. Разбиение Penn-Fudan: 136 / 17 / 17, seed=42.

В `results/` сохранены модели, история обучения, метрики и рисунки. При `RETRAIN = False` ноутбук использует сохранённые модели; при `True` обучает базовые модели заново. Для результатов в отчёте модель CARLA обучалась 40 эпох, модель Penn-Fudan — 20 эпох. Подготовка и обучение могут занимать время на CPU.

Для обновления отчёта по сохранённым результатам:

```bash
python make_report.py
```

## Результат

На тестовой части CARLA базовая модель получила accuracy 0,883 и mean IoU 0,260. На 17 тестовых изображениях Penn-Fudan получены accuracy 0,897, mean IoU 0,693 и IoU класса «человек» 0,500. Это разные задачи, поэтому их метрики напрямую не сравниваются. Для собственных фотографий масок нет; на них показаны только предсказания.
