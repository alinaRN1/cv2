"""Краткий отчёт из результатов двух реально выполненных экспериментов."""

import csv
import json
from pathlib import Path

import numpy as np


def read_result(name, kind):
    return json.loads(Path(f"results/{name}_{kind}.json").read_text())


def write_report():
    text = ["# Лабораторная работа № 2", "",
            "## Цель и метод", "",
            "Построить U-Net для сегментации изображений и повторить обучение на двух наборах: "
            "учебном CARLA и реальном CamVid. Encoder содержит пять блоков, decoder — четыре. "
            "Между ними есть skip connections. Использованы Adam и sparse categorical "
            "crossentropy по logits, batch size 8 и 40 эпох. Вход — 96×128×3. "
            "Сначала рассмотрена модель с 8 начальными фильтрами, затем — модель "
            "с 32 фильтрами и взвешенной функцией потерь.", "",
            "## Данные", "",
            "CARLA: 1060 изображений и соответствующих масок для 23 классов. "
            "[Источник набора данных](https://github.com/Rhythm-Adhikari1/U-Net/tree/072afa362ab567ceb8c151b18b00e05ac632bf26/data). "
            "Разбиение 848/106/106 выполнено целыми группами близких номеров "
            "кадров. CamVid: 701 реальное изображение, 11 смысловых классов и Void, "
            "сохранено опубликованное разбиение 367/101/233. Маски приведены к размеру "
            "входа через nearest neighbor. В CARLA ID читается из красного канала, "
            "в CamVid Void исключён из loss и оценки. Источники и версии указаны в README; "
            "контрольные суммы — в results/data_manifest.json.", "",
            "## Итоговые результаты", "",
            "Лучшая модель выбрана по validation loss. Baseline предсказывает только "
            "самый частый класс train. Тест не использовался для выбора эпохи.", "",
            "| Набор | Тестовых кадров | Accuracy | Baseline accuracy | Mean IoU | Baseline mIoU | Лучшая эпоха |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ["carla", "camvid"]:
        scores, run = read_result(name, "metrics"), read_result(name, "run")
        baseline = scores["majority_baseline"]
        text.append(f"| {name.upper()} | {scores['test_images']} | {scores['accuracy']:.4f} | "
                    f"{baseline['accuracy']:.4f} | {scores['mean_iou']:.4f} | "
                    f"{baseline['mean_iou']:.4f} | {run['best_epoch']} |")
    for name in ["carla", "camvid"]:
        scores, stats, run = [read_result(name, kind) for kind in ["metrics", "data", "run"]]
        classes = scores["class_names"]
        history = list(csv.DictReader(Path(f"results/{name}_history.csv").open()))
        best = history[run["best_epoch"] - 1]
        last = history[-1]
        if run["best_epoch"] < run["epochs"] and float(last["val_loss"]) > float(best["val_loss"]) * 1.05:
            trend = ("К концу обучения validation loss ухудшилась при меньшей train loss — "
                     "это признак переобучения. Для теста использована более ранняя эпоха. ")
        else:
            trend = "В этом запуске минимальная validation loss достигнута на последней эпохе. "
        majority = stats["majority_class"]
        text += ["", "## " + name.upper(), "",
                 f"Обучение выполнено на всех {run['split_sizes']['train']} обучающих кадрах, "
                 f"{run['epochs']} эпох. У модели {run['parameters']:,} параметров. "
                 f"На выбранной эпохе train loss = {float(best['loss']):.4f}, "
                 f"validation loss = {float(best['val_loss']):.4f}. "
                 f"На последней эпохе train loss = {float(last['loss']):.4f}, "
                 f"validation loss = {float(last['val_loss']):.4f}. "
                 + trend +
                 f"Baseline использует класс {classes[majority]}. "
                 f"Превышение baseline: {(scores['accuracy'] - scores['majority_baseline']['accuracy']) * 100:.2f} "
                 f"процентного пункта accuracy и {scores['mean_iou'] - scores['majority_baseline']['mean_iou']:.4f} mean IoU.", "",
                 f"![Баланс классов](results/{name}_classes.png)", "",
                 f"![Обучение](results/{name}_history.png)", "",
                 "| Класс | Оцениваемых пикселей | IoU |", "| --- | ---: | ---: |"]
        for label, count, score in zip(classes, scores["class_pixels"], scores["iou_per_class"]):
            value = f"{score:.4f}" if score is not None else "не оценивается"
            text.append(f"| {label} | {count} | {value} |")
        text += ["", "Самые частые ошибки по числу пикселей:", ""]
        confusion = np.array(scores["confusion_matrix"])
        np.fill_diagonal(confusion, 0)
        errors = sorted(zip(*np.nonzero(confusion)), key=lambda ij: confusion[ij], reverse=True)[:5]
        text += [f"- {classes[true]} → {classes[pred]}: {confusion[true, pred]} пикселей."
                 for true, pred in errors]
        low = sorted((value, classes[i]) for i, value in enumerate(scores["iou_per_class"])
                     if value is not None and scores["class_pixels"][i] > 0)[:3]
        text += ["", "Наименьшие IoU среди присутствующих классов: " +
                 ", ".join(f"{label} ({value:.3f})" for value, label in low) + ". "
                 "Общая accuracy скрывает часть этих ошибок из-за дисбаланса числа пикселей."]
        missed = [classes[i] for i, value in enumerate(scores["iou_per_class"])
                  if value == 0 and scores["class_pixels"][i] > 0]
        if missed:
            text += ["", "У классов " + ", ".join(missed) +
                     " IoU равен нулю при наличии тестовых пикселей. "
                     "Их распознавание в этом эксперименте не получилось, несмотря на общую accuracy."]
    text += ["", "## Проверки и ограничения", "",
             "Проверены размеры выходов сети, предобработка изображений и масок, "
             "расчёт IoU и сохранение модели. Также проверено отсутствие одинаковых "
             "изображений между частями данных и разделение близких кадров CARLA.", "",
             "Результаты получены при одном seed=42. Группировка CARLA по ID уменьшает "
             "утечку соседних кадров, но не гарантирует независимость поездок без "
             "метаданных. CamVid использует стандартный тест, однако сцены внутри наборов "
             "могут иметь сходные условия. Низкое разрешение и малая ширина сети ограничивают "
             "качество мелких классов. Прямое сравнение чисел CARLA и CamVid некорректно: "
             "различаются классы, источники и разбиения.", "",
             "## Вывод", "",
             "Реализован и выполнен полный процесс сегментации: подготовка двух наборов, "
             "построение U-Net, обучение, выбор модели по валидации и независимая тестовая "
             "оценка. Поклассовый IoU и сравнение с baseline позволяют увидеть ограничения, "
             "которые не показывает одна accuracy. Следующий шаг для повышения качества — "
             "увеличить разрешение и проверить балансировку функции потерь на валидации."]
    report = "\n".join(text) + "\n"
    folder = Path("results/experiment_32")
    if all((folder / f"{name}_comparison.json").exists() for name in ["carla", "camvid"]):
        from update_work import report_appendix
        report = report.replace(
            "Следующий шаг для повышения качества — увеличить разрешение и проверить балансировку функции потерь на валидации.",
            "Для дальнейшей проверки нужны новые тестовые кадры и повторения с другими seed.")
        report += "\n" + report_appendix() + "\n"
    Path("REPORT.md").write_text(report)


if __name__ == "__main__":
    write_report()
