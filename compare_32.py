"""Сравнение прежней и новой моделей: выбор по validation, оценка по test."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cv2-matplotlib")
if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lab2 import compile_model, evaluate_model, show_predictions, worst_pairs
from prepare_data import load_split, check_manifest, CARLA_CLASSES, CAMVID_CLASSES


def evaluate(name):
    check_manifest()
    directory = Path("results/experiment_32")
    classes = CARLA_CLASSES if name == "carla" else CAMVID_CLASSES
    ignore = None if name == "carla" else 11
    split = load_split(name)
    config = json.loads((directory / f"{name}_config.json").read_text())
    with (directory / f"{name}_history.csv").open() as file:
        rows = list(csv.DictReader(file))
    if len(rows) != config["epochs"] or [int(row["epoch"]) for row in rows] != list(range(config["epochs"])):
        raise ValueError("История обучения неполная или повреждена")

    # Compile=False: метрики считаются одинаково для двух моделей, без весов loss.
    old = tf.keras.models.load_model(f"results/{name}.keras", compile=False)
    new = tf.keras.models.load_model(directory / f"{name}_best.keras", compile=False)
    compile_model(old, ignore)
    compile_model(new, ignore)
    majority = json.loads(Path(f"results/{name}_data.json").read_text())["majority_class"]
    old_val_path = directory / f"{name}_old_val_metrics.json"
    if old_val_path.exists():
        old_val = json.loads(old_val_path.read_text())
        if old_val["test_images"] != len(split["val"]):
            raise ValueError("Результаты прежней модели относятся к другому разбиению")
    else:
        old_val = evaluate_model(old, split["val"], (96, 128), len(classes),
                                 f"experiment_32/{name}_old_val", ignore, majority, classes)
    new_val = evaluate_model(new, split["val"], (96, 128), len(classes),
                             f"experiment_32/{name}_new_val", ignore, majority, classes)
    best_epoch = int(np.argmax([float(row["val_mean_iou"]) for row in rows])) + 1
    if abs(float(rows[best_epoch-1]["val_mean_iou"]) - new_val["mean_iou"]) > 0.002:
        raise ValueError("Загруженные веса не соответствуют лучшей валидационной эпохе")

    # Критерий выбирается до повторной оценки на тестовой части.
    selected = "32" if new_val["mean_iou"] > old_val["mean_iou"] else "8"
    old_test = json.loads(Path(f"results/{name}_metrics.json").read_text())
    new_test = evaluate_model(new, split["test"], (96, 128), len(classes),
                              f"experiment_32/{name}_new_test", ignore, majority, classes)
    if new_test["class_pixels"] != old_test["class_pixels"]:
        raise ValueError("Модели оценены на разных тестовых пикселях")
    show_predictions(new, split["test"], (96, 128), len(classes),
                     f"experiment_32/{name}", class_names=classes, ignore_label=ignore)
    show_predictions(new, worst_pairs(new_test, split["test"]), (96, 128), len(classes),
                     f"experiment_32/{name}_worst", count=2, class_names=classes,
                     ignore_label=ignore)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    epoch = np.arange(1, len(rows) + 1)
    for ax, metric, title in zip(axes, ["loss", "mean_iou"], ["Weighted loss", "Mean IoU"]):
        ax.plot(epoch, [float(row[metric]) for row in rows], label="train")
        ax.plot(epoch, [float(row["val_" + metric]) for row in rows], label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(directory / f"{name}_curves.png", dpi=140)
    plt.close("all")

    clean = tf.keras.Model.from_config(new.get_config())
    clean.set_weights(new.get_weights())
    clean.save(directory / f"{name}_inference.keras")
    result = {"dataset": name, "selection_metric": "validation mean IoU",
              "selected_filters": selected, "best_epoch_32": best_epoch,
              "validation_mean_iou": {"8": old_val["mean_iou"], "32": new_val["mean_iou"]},
              "test_mean_iou": {"8": old_test["mean_iou"], "32": new_test["mean_iou"]},
              "test_accuracy": {"8": old_test["accuracy"], "32": new_test["accuracy"]},
              "test_iou_per_class": {"8": old_test["iou_per_class"],
                                     "32": new_test["iou_per_class"]},
              "classes": classes, "test_images": len(split["test"])}
    (directory / f"{name}_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["carla", "camvid"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Число потоков должно быть положительным")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, args.threads))
    evaluate(args.dataset)
