"""Добавить результаты дополнительного эксперимента в notebook и отчёт."""

import json
from pathlib import Path

import nbformat

DIRECTORY = Path("results/experiment_32")
HEADING = "## 7. Дополнительный эксперимент с полной шириной U-Net"


def read_comparison(name):
    return json.loads((DIRECTORY / f"{name}_comparison.json").read_text())


def comparison_table():
    lines = ["| Набор | Validation IoU, 8 фильтров | Validation IoU, 32 фильтра | "
             "Test IoU, 8 фильтров | Test IoU, 32 фильтра | Выбор по validation |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ["carla", "camvid"]:
        result = read_comparison(name)
        v, t = result["validation_mean_iou"], result["test_mean_iou"]
        lines.append(f"| {name.upper()} | {v['8']:.3f} | {v['32']:.3f} | "
                     f"{t['8']:.3f} | {t['32']:.3f} | {result['selected_filters']} |")
    return "\n".join(lines)


def external_table():
    result = json.loads((DIRECTORY / "carla_external_metrics.json").read_text())
    lines = ["| Модель | Pixel accuracy | Mean IoU по классам 1–12 |",
             "| --- | ---: | ---: |"]
    for label, metrics in [("Baseline: Road", result["baseline_road"]),
                           ("U-Net, 8 фильтров", result["models"]["8"]),
                           ("U-Net, 32 фильтра", result["models"]["32"])]:
        lines.append(f"| {label} | {metrics['accuracy']:.3f} | {metrics['mean_iou']:.3f} |")
    return "\n".join(lines)


def mean_iou_present(ious, class_pixels):
    indices = [i for i, count in enumerate(class_pixels) if count > 0]
    if not indices or any(ious[i] is None for i in indices):
        raise ValueError("Недостаточно данных для среднего по присутствующим классам")
    return sum(ious[i] for i in indices) / len(indices)


def present_classes_table():
    lines = ["| Набор | Классов в разметке | mIoU: 8 фильтров | mIoU: 32 фильтра |",
             "| --- | ---: | ---: | ---: |"]
    for name in ["carla", "camvid"]:
        result = read_comparison(name)
        metrics = json.loads((DIRECTORY / f"{name}_new_test_metrics.json").read_text())
        present = [i for i, count in enumerate(metrics["class_pixels"]) if count > 0]
        means = {width: mean_iou_present(result["test_iou_per_class"][width], metrics["class_pixels"])
                 for width in ["8", "32"]}
        lines.append(f"| {name.upper()} | {len(present)} | {means['8']:.3f} | {means['32']:.3f} |")
    return "\n".join(lines)


def report_appendix():
    external_ready = (DIRECTORY / "carla_external_metrics.json").exists()
    lines = [HEADING, "",
             "В дополнительном эксперименте использована полная ширина сети из условия: "
             "32, 64, 128, 256 и 512 фильтров в encoder. Обе модели обучены заново, "
             "по 40 эпох. Для уменьшения влияния дисбаланса классов веса функции потерь "
             "рассчитаны только по train: w_c пропорционален 1 / sqrt(max(f_c, 0.002)), "
             "где f_c — доля пикселей класса; затем веса нормированы так, чтобы "
             "средний вес обучающего пикселя был равен единице. "
             "У отсутствующих в train классов вес равен нулю. "
             "Void CamVid исключён из обучения и оценки. Контрольная точка выбрана по "
             "максимальному mean IoU на validation; тест для этого не использован.", "",
             comparison_table(), "",
             "В стандартном mIoU класс с нулевым объединением исключается, а отсутствующий "
             "в разметке, но ошибочно предсказанный класс получает IoU=0. Поэтому число "
             "усредняемых классов у моделей может различаться. Дополнительно привожу "
             "среднее только по классам, присутствующим в тестовой разметке. Ошибочные "
             "предсказания отсутствующих классов при этом снижают IoU истинных классов.", "",
             present_classes_table(), "",
             "Обе версии используют одинаковые данные, разрешение и разбиения. "
             "Одновременно изменились ширина сети и функция потерь, поэтому определить "
             "отдельный вклад каждого изменения нельзя. Их loss нельзя сравнивать напрямую. "
             "Модель с 8 фильтрами выбрана по минимальному validation loss, модель с 32 — "
             "по максимальному validation IoU; схемы выбора эпох различаются. "
             "Тестовые результаты модели с 8 фильтрами анализировались при выборе "
             "следующего эксперимента, поэтому сравнение носит исследовательский характер. "
             + ("Дополнительная проверка на новых кадрах описана в разделе 8."
                if external_ready else "Для независимой оценки нужен новый набор кадров."), ""]
    for name in ["carla", "camvid"]:
        result = read_comparison(name)
        lines += [f"### {name.upper()}: полная сеть", "",
                  f"Лучшая эпоха по validation IoU: {result['best_epoch_32']}. "
                  f"Accuracy на тесте: {result['test_accuracy']['32']:.3f}; "
                  f"mean IoU: {result['test_mean_iou']['32']:.3f}.", "",
                  f"![Кривые {name}](results/experiment_32/{name}_curves.png)", "",
                  "| Класс | IoU: 8 фильтров | IoU: 32 фильтра |",
                  "| --- | ---: | ---: |"]
        for label, old, new in zip(result["classes"], result["test_iou_per_class"]["8"],
                                    result["test_iou_per_class"]["32"]):
            fmt = lambda x: f"{x:.3f}" if x is not None else "—"
            lines.append(f"| {label} | {fmt(old)} | {fmt(new)} |")
        metrics = json.loads((DIRECTORY / f"{name}_new_test_metrics.json").read_text())
        changes = [(new - old, label) for label, old, new, pixels in zip(
            result["classes"], result["test_iou_per_class"]["8"],
            result["test_iou_per_class"]["32"], metrics["class_pixels"])
            if old is not None and new is not None and pixels > 0]
        gains = sorted((d, label) for d, label in changes if d > 0)[-3:][::-1]
        losses = sorted((d, label) for d, label in changes if d < 0)[:3]
        if gains:
            lines += ["", "Наибольший рост IoU: " +
                      ", ".join(f"{label} ({delta:+.3f})" for delta, label in gains) + "."]
        if losses:
            lines += ["", "Наибольшее снижение IoU: " +
                      ", ".join(f"{label} ({delta:+.3f})" for delta, label in losses) + "."]
        lines += ["", f"![Предсказания {name}](results/experiment_32/{name}_predictions.png)",
                  "", f"![Трудные примеры {name}](results/experiment_32/{name}_worst_predictions.png)", ""]
        matrix = metrics["confusion_matrix"]
        errors = sorted(((matrix[true][pred], true, pred)
                         for true in range(len(matrix)) for pred in range(len(matrix))
                         if true != pred and matrix[true][pred]), reverse=True)[:3]
        if errors:
            lines += ["Самые частые ошибки по числу пикселей:", ""]
            lines += [f"- {result['classes'][true]} → {result['classes'][pred]}: {count}."
                      for count, true, pred in errors]
            lines.append("")
        absent = [(sum(row[i] for row in matrix), label)
                  for i, label in enumerate(result["classes"])
                  if metrics["class_pixels"][i] == 0 and sum(row[i] for row in matrix) > 0]
        if absent:
            lines += ["Ошибочные предсказания классов, отсутствующих среди оцениваемых "
                      "тестовых пикселей: " +
                      ", ".join(f"{label} ({count} пикселей)" for count, label in
                                sorted(absent, reverse=True)) + ".", ""]
        zero = [label for label, value in zip(result["classes"], result["test_iou_per_class"]["32"])
                if value == 0]
        if zero:
            lines += ["Классы с нулевым IoU в этой оценке: " + ", ".join(zero) +
                      ". Их качество требует дальнейшей работы.", ""]
    external_path = DIRECTORY / "carla_external_metrics.json"
    if external_ready:
        result = json.loads(external_path.read_text())
        selected = result["selection_from_validation"]
        valid_pixels = result["models"]["8"]["evaluated_pixels"]
        total_pixels = result["images"] * 96 * 128
        lines += ["## 8. Проверка на другом наборе CARLA", "",
                  "Для независимой проверки взяты 223 контрольных кадра из "
                  "[отдельного набора CARLA 0.9.5](https://github.com/cpow-89/carla_image_segmentation_dataset/tree/aa0eb00150f3fda6f7950f097baa145890717979). "
                  "Они не использовались при обучении, выборе весов или параметров. "
                  "Совпадений файлов с предыдущими наборами по SHA-256 нет. "
                  "Метка 0 означает Unlabeled и исключена; IoU усредняется по "
                  "поддерживаемым классам 1–12, присутствующим в разметке: "
                  "один и тот же набор классов используется для всех моделей. "
                  "Предсказание других классов на размеченных пикселях считается ошибкой. "
                  "Код 10 называется Car в этом наборе и Vehicle в учебном. "
                  "Сравнивать это значение напрямую с mIoU по 23 классам нельзя. "
                  "На иллюстрациях неразмеченные области замаскированы и в предсказаниях. "
                  f"Оценено {valid_pixels} из {total_pixels} пикселей после resize "
                  f"({valid_pixels / total_pixels:.1%}); остальные имеют метку Unlabeled.", "",
                  external_table(), "",
                  f"По прежней validation-части выбрана модель с {selected} начальными фильтрами. "
                  "Контрольные кадры получены отдельно, но разные наборы используют "
                  "симулятор CARLA и могут содержать общие виртуальные сцены. "
                  "Кадры внутри одного run также могут быть похожи между собой, "
                  "поэтому 223 кадра не равнозначны 223 независимым сценам.", "",
                  "| Класс | Пиксели | IoU: 8 фильтров | IoU: 32 фильтра |",
                  "| --- | ---: | ---: | ---: |"]
        carla_classes = json.loads(Path("results/carla_data.json").read_text())["class_names"]
        for label in range(1, 13):
            old = result["models"]["8"]["iou_per_class"][label]
            new = result["models"]["32"]["iou_per_class"][label]
            count = result["models"]["8"]["class_pixels"][label]
            fmt = lambda value: f"{value:.3f}" if value is not None else "—"
            lines.append(f"| {carla_classes[label]} | {count} | {fmt(old)} | {fmt(new)} |")
        if result["models"]["8"]["class_pixels"][4] == 0:
            false_old = sum(row[4] for row in result["models"]["8"]["confusion_matrix"])
            false_new = sum(row[4] for row in result["models"]["32"]["confusion_matrix"])
            lines += ["", "В этой контрольной части нет размеченных пешеходов, "
                      "поэтому recall класса Pedestrian по ней определить нельзя. "
                      f"На остальных размеченных пикселях модель с 8 фильтрами ошибочно "
                      f"предсказала Pedestrian {false_old} раз, с 32 фильтрами — {false_new} раз."]
        lines += ["", "![Независимые кадры](results/experiment_32/carla_external_predictions.png)", ""]
        lines += ["### Вывод по дополнительному эксперименту", ""]
        for name in ["carla", "camvid"]:
            choice = read_comparison(name)
            width = choice["selected_filters"]
            lines.append(f"- Для {name.upper()} по validation выбрана модель с {width} фильтрами "
                         f"(mIoU {choice['validation_mean_iou'][width]:.3f}).")
        lines += ["", f"На отдельных кадрах CARLA выбранная дорожная модель получила "
                  f"mIoU {result['models'][selected]['mean_iou']:.3f} по классам 1–12; "
                  f"baseline получил {result['baseline_road']['mean_iou']:.3f}. "
                  "Результат описывает только оцениваемые классы и условия этого набора. "
                  "Для устойчивых выводов нужны другие сцены и повторения обучения с разными seed.", ""]
    lines += ["", "## 9. Проверка на собственных фотографиях", "",
              "Для дополнительной визуальной проверки использованы четыре собственные "
              "фотографии: город, набережная и два горных дорожных вида. Каждый снимок "
              "приведён к 96×128 и обработан уже обученными моделями с 8 и 32 фильтрами. "
              "Цвета показывают предсказанные классы; разметки этих фотографий нет, "
              "поэтому accuracy и IoU для них не вычислялись.", "",
              "![Фотографии и предсказания CARLA](results/carla_own_photos.png)", "",
              "![Фотографии и предсказания CamVid](results/camvid_own_photos.png)", "",
              "На городских снимках обе модели выделяют отдельные крупные области, но "
              "ошибаются на мелких объектах и границах. Горные сцены отличаются от "
              "дорожных кадров обучения: местность и другие детали нередко получают "
              "неподходящие дорожные классы. Это качественная проверка переноса, "
              "а приведённые выше численные метрики относятся к размеченным тестовым наборам."]
    return "\n".join(lines)


def update():
    for name in ["carla", "camvid"]:
        if not (DIRECTORY / f"{name}_comparison.json").is_file():
            raise FileNotFoundError(f"Сначала завершите сравнение {name}")
    notebook_path = Path("CV_L2.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            note = ("Имена первых файлов и пример `002128.png` совпадают с приложенным notebook; "
                    "полное побайтовое сравнение с недоступным Google Drive не проводилось.")
            cell.source = cell.source.replace("\n\n" + note, "")
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                "RETRAIN = False  # True — заново обучить обе модели; False — использовать готовые при наличии.",
                "RETRAIN = False  # True — заново обучить обе модели с 8 фильтрами.")
            cell.source = cell.source.replace(
                '"camvid", class_names=CAMVID_CLASSES)',
                '"camvid", class_names=CAMVID_CLASSES, ignore_label=11)')
            cell.source = cell.source.replace(
                'name.lower() + "_worst", count=2, class_names=classes)',
                'name.lower() + "_worst", count=2, class_names=classes, '
                'ignore_label=11 if name == "CamVid" else None)')
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith(HEADING):
            notebook.cells = notebook.cells[:i]
            break
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        HEADING + "\n\n" +
        "Использую 32 начальных фильтра и 40 эпох, как в исходном примере. "
        "Редкие классы учитываются взвешенной crossentropy; веса вычислены по train. "
        "Лучшая эпоха выбирается по mean IoU на validation. Ниже загружаются результаты "
        "обучения, сохранённые скриптами `experiment_32.py` и `compare_32.py`."))
    notebook.cells.append(nbformat.v4.new_code_cell('''from IPython.display import Image
from update_work import comparison_table, present_classes_table, read_comparison
display(Markdown(comparison_table()))
display(Markdown("Средний IoU только по классам, присутствующим в тестовой разметке:\\n\\n" + present_classes_table()))
for name in ["carla", "camvid"]:
    result = read_comparison(name)
    metrics = json.loads(Path(f"results/experiment_32/{name}_new_test_metrics.json").read_text())
    rows = ["| Класс | Пиксели | IoU: 8 фильтров | IoU: 32 фильтра |", "| --- | ---: | ---: | ---: |"]
    for label, pixels, old, new in zip(result["classes"], metrics["class_pixels"],
                                       result["test_iou_per_class"]["8"], result["test_iou_per_class"]["32"]):
        old_text = f"{old:.3f}" if old is not None else "—"
        new_text = f"{new:.3f}" if new is not None else "—"
        rows.append(f"| {label} | {pixels} | {old_text} | {new_text} |")
    display(Markdown("### " + name.upper() + "\\n\\n" + "\\n".join(rows)))
    for kind in ["curves", "predictions", "worst_predictions"]:
        display(Image(filename=f"results/experiment_32/{name}_{kind}.png"))
'''))
    independent_note = ("Дополнительная проверка на новых кадрах приведена ниже."
                        if (DIRECTORY / "carla_external_metrics.json").exists()
                        else "Для независимой итоговой оценки нужны новые тестовые кадры.")
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "Тестовый набор использовался при анализе модели с 8 фильтрами, поэтому "
        "сравнение на нём носит исследовательский характер. " + independent_note))
    if (DIRECTORY / "carla_external_metrics.json").exists():
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            "## 8. Проверка на другом наборе CARLA\n\n"
            "Контрольные 223 кадра из другого опубликованного набора не участвовали "
            "в обучении и выборе модели. В этой версии CARLA поддерживаются классы "
            "1–12; Unlabeled=0 исключается, а ошибки предсказания других классов "
            "учитываются. На иллюстрациях исключённые области замаскированы. "
            "Пешеходов в контрольной части нет. Данные и ограничения "
            "сравнения описаны в отчёте."))
        notebook.cells.append(nbformat.v4.new_code_cell('''from update_work import external_table
display(Markdown(external_table()))
external = json.loads(Path("results/experiment_32/carla_external_metrics.json").read_text())
valid_pixels = external["models"]["8"]["evaluated_pixels"]
print("Оценено пикселей:", valid_pixels, "из", external["images"] * 96 * 128)
rows = ["| Класс | Пиксели | IoU: 8 фильтров | IoU: 32 фильтра |",
        "| --- | ---: | ---: | ---: |"]
for label in range(1, 13):
    old = external["models"]["8"]["iou_per_class"][label]
    new = external["models"]["32"]["iou_per_class"][label]
    pixels = external["models"]["8"]["class_pixels"][label]
    old_text = f"{old:.3f}" if old is not None else "—"
    new_text = f"{new:.3f}" if new is not None else "—"
    rows.append(f"| {CARLA_CLASSES[label]} | {pixels} | {old_text} | {new_text} |")
display(Markdown("\\n".join(rows)))
for width in ["8", "32"]:
    false_pedestrian = sum(row[4] for row in external["models"][width]["confusion_matrix"])
    print(f"Ошибочные предсказания пешеходов, {width} фильтров: {false_pedestrian} пикселей")
display(Image(filename="results/experiment_32/carla_external_predictions.png"))
'''))
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        "## 9. Собственные фотографии\n\n"
        "Для визуальной проверки использую четыре собственные фотографии. "
        "Показываю предсказания обеих обученных моделей после изменения размера "
        "до 96×128. Масок для этих фотографий нет, поэтому IoU на них не вычисляется. "
        "Горные виды отличаются от дорожных сцен обучения; мелкие объекты и границы "
        "часто распознаются неверно."))
    notebook.cells.append(nbformat.v4.new_code_cell('''from own_photos import show_own_photos
for dataset in ["carla", "camvid"]:
    print(dataset.upper())
    _ = show_own_photos(dataset)
'''))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.write(notebook, notebook_path)

    report_path = Path("REPORT.md")
    report = report_path.read_text()
    if HEADING in report:
        report = report[:report.index(HEADING)].rstrip() + "\n"
    report = report.replace(
        "Следующий шаг для повышения качества — увеличить разрешение и проверить балансировку функции потерь на валидации.",
        "Для дальнейшей проверки нужны новые тестовые кадры и повторения с другими seed.")
    report_path.write_text(report.rstrip() + "\n\n" + report_appendix() + "\n")

    readme_path = Path("README.md")
    readme = readme_path.read_text()
    readme = readme.replace(
        "Исходная ширина 32 доступна через\n`n_filters=32` и проверяется тестами, но её полное обучение на этой машине по замеру\nзаняло бы около 2–3 часов.",
        "Исходная ширина 32 доступна через `n_filters=32`; дополнительные полные\nобучения и сравнение описаны ниже.")
    readme = readme.replace("обе обученные модели `*.keras`", "обученные модели `*.keras`")
    readme = readme.replace("python prepare_data.py\n```",
                            "python prepare_data.py\npython external_carla.py prepare\n```")
    readme = readme.replace("Для нового обучения выставить\n`RETRAIN=True` в первой кодовой ячейке.",
                            "Для нового обучения моделей с 8 фильтрами выставить\n`RETRAIN=True` в первой кодовой ячейке. Для моделей с 32 фильтрами\nиспользуются отдельные команды ниже.")
    marker = "## Полная сеть и дополнительная проверка"
    if marker in readme:
        readme = readme[:readme.index(marker)].rstrip() + "\n"
    readme += '''

## Полная сеть и дополнительная проверка

В `results/experiment_32/` сохранены результаты двух дополнительных обучений
с 32 начальными фильтрами, по 40 эпох. Классы взвешены по частотам только в train;
лучшая эпоха выбирается по validation mean IoU. Предыдущие модели на 8 фильтрах
сохранены для сравнения. `*_inference.keras` содержит обученную модель без
состояния оптимизатора, а `*_history.csv` и `*_config.json` — историю и параметры.

Дополнительная контрольная часть содержит 223 кадра из другого публичного набора
CARLA 0.9.5. Она не использовалась при обучении и выборе модели. Метка 0
(`Unlabeled`) исключена из оценки, средний IoU вычисляется по классам 1–12,
присутствующим в разметке. Это иная схема, чем 23 класса исходного набора;
ошибки других предсказанных классов на размеченных пикселях учитываются.
Наборы созданы независимо, но могут содержать общие виртуальные сцены CARLA.

Полное повторение занимает продолжительное время на CPU:

```bash
python external_carla.py prepare
python experiment_32.py carla
python experiment_32.py camvid
python compare_32.py carla
python compare_32.py camvid
python external_carla.py evaluate
python make_report.py
python update_work.py
jupyter-execute CV_L2.ipynb --inplace --timeout 7200
python verify_results.py
```

`external_carla.py prepare` скачивает только 223 пары опубликованной контрольной
части по закреплённому Git-коммиту, проверяет формат масок и совпадение SHA-256
с ранее использованными изображениями. Данные хранятся в `data/carla_external/`;
в архиве есть контрольные суммы, исходные изображения скачиваются отдельно.
Повторный запуск проверки сверяет файлы с сохранённым манифестом.

Тестовые результаты модели с 8 фильтрами учитывались при выборе следующего
эксперимента, поэтому сравнение на том же тесте носит исследовательский характер.
Дополнительные 223 кадра используются после выбора модели по validation. Для
новых научных выводов полезно также повторить обучение с несколькими seed и
изучить реальные изображения других городов.
'''
    readme_path.write_text(readme)


if __name__ == "__main__":
    update()
