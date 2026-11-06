"""Проверка, что notebook, история, метрики и сохранённые веса относятся к полному запуску."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cv2-matplotlib")

import nbformat
import numpy as np
import tensorflow as tf

from lab2 import read_pair, scores_from_confusion
from prepare_data import load_split


def verify():
    notebook = nbformat.read("CV_L2.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    validation = {"executed_code_cells": len(code_cells), "experiments": {}}
    for name, expected in [("carla", 106), ("camvid", 233)]:
        def read(kind):
            return json.loads(Path(f"results/{name}_{kind}.json").read_text())
        scores, run = read("metrics"), read("run")
        rows = list(csv.DictReader(Path(f"results/{name}_history.csv").open()))
        assert len(rows) == run["epochs"] == 40
        assert [int(row["epoch"]) for row in rows] == list(range(40))
        assert all(np.isfinite(float(row[key])) for row in rows
                   for key in ["loss", "accuracy", "val_loss", "val_accuracy"])
        best_epoch = int(np.argmin([float(row["val_loss"]) for row in rows])) + 1
        assert best_epoch == run["best_epoch"]
        assert scores["test_images"] == run["split_sizes"]["test"] == expected
        recalculated = scores_from_confusion(scores["confusion_matrix"], run["ignore_label"])
        for key in ["accuracy", "mean_iou", "evaluated_pixels"]:
            assert abs(scores[key] - recalculated[key]) < 1e-8
        model = tf.keras.models.load_model(f"results/{name}.keras", compile=False)
        assert model.count_params() == run["parameters"]
        assert model.output_shape == (None, *run["size"], run["n_classes"])
        differences = []
        expected_per_image = {item["image"]: item["mean_iou"] for item in scores["per_image"]}
        pairs = load_split(name)["test"]
        for pair in [pairs[0], pairs[len(pairs) // 2], pairs[-1]]:
            x, y = read_pair(*pair, run["size"], run["n_classes"])
            logits = model(x[None], training=False).numpy()
            assert np.isfinite(logits).all()
            pred = logits.argmax(axis=-1)
            matrix = tf.math.confusion_matrix(y.ravel(), pred.ravel(),
                                              num_classes=run["n_classes"]).numpy()
            actual = scores_from_confusion(matrix, run["ignore_label"])["mean_iou"]
            difference = abs(actual - expected_per_image[str(pair[0])])
            assert difference < 1e-3, (name, pair[0], difference)
            differences.append(difference)
        validation["experiments"][name] = {
            "epochs": len(rows), "best_epoch": best_epoch,
            "test_images": expected, "saved_model_checked_on_images": 3,
            "maximum_iou_difference_after_reload": max(differences)}
    directory = Path("results/experiment_32")
    comparison_files = [directory / f"{name}_comparison.json" for name in ["carla", "camvid"]]
    experiment_started = any((directory / f"{name}_config.json").exists() for name in ["carla", "camvid"])
    if experiment_started and not all(path.exists() for path in comparison_files):
        raise AssertionError("Дополнительное обучение или сравнение двух наборов ещё не завершено")
    if all(path.exists() for path in comparison_files):
        validation["experiments_32"] = {}
        for name in ["carla", "camvid"]:
            config = json.loads((directory / f"{name}_config.json").read_text())
            comparison = json.loads((directory / f"{name}_comparison.json").read_text())
            test = json.loads((directory / f"{name}_new_test_metrics.json").read_text())
            old_test = json.loads(Path(f"results/{name}_metrics.json").read_text())
            train_counts = np.array(json.loads(Path(f"results/{name}_data.json").read_text())
                                    ["train"]["class_pixels"], dtype=np.int64)
            if config["ignore_label"] is not None:
                train_counts[config["ignore_label"]] = 0
            assert np.array_equal(train_counts, np.array(config["class_counts_train"]))
            class_weights = np.array(config["class_weights"])
            assert len(class_weights) == config["n_classes"]
            assert np.all(np.isfinite(class_weights)) and np.all(class_weights >= 0)
            assert np.all(class_weights[train_counts == 0] == 0)
            assert np.isclose(np.dot(class_weights, train_counts / train_counts.sum()), 1)
            with (directory / f"{name}_history.csv").open() as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == config["epochs"] == 40
            assert [int(row["epoch"]) for row in rows] == list(range(40))
            best = int(np.argmax([float(row["val_mean_iou"]) for row in rows])) + 1
            assert best == comparison["best_epoch_32"]
            expected_choice = "32" if comparison["validation_mean_iou"]["32"] > comparison["validation_mean_iou"]["8"] else "8"
            assert comparison["selected_filters"] == expected_choice
            assert abs(test["mean_iou"] - comparison["test_mean_iou"]["32"]) < 1e-8
            assert abs(old_test["mean_iou"] - comparison["test_mean_iou"]["8"]) < 1e-3
            assert test["class_pixels"] == old_test["class_pixels"]
            for width, kind in [("8", "old"), ("32", "new")]:
                val_scores = json.loads((directory / f"{name}_{kind}_val_metrics.json").read_text())
                assert abs(val_scores["mean_iou"] - comparison["validation_mean_iou"][width]) < 1e-8
            model = tf.keras.models.load_model(directory / f"{name}_inference.keras", compile=False)
            assert model.output_shape == (None, *config["size"], config["n_classes"])
            assert model.count_params() > 8_000_000
            pairs = load_split(name)["test"][:8]
            samples = [read_pair(*pair, config["size"], config["n_classes"]) for pair in pairs]
            images = np.stack([sample[0] for sample in samples])
            logits = model(images, training=False).numpy()
            expected = {item["image"]: item["mean_iou"] for item in test["per_image"]}
            differences = []
            for pair, (_, y), pred in zip(pairs[:3], samples[:3], logits[:3].argmax(axis=-1)):
                cm = tf.math.confusion_matrix(y.ravel(), pred.ravel(),
                                              num_classes=config["n_classes"]).numpy()
                actual = scores_from_confusion(cm, config["ignore_label"])["mean_iou"]
                differences.append(abs(actual - expected[str(pair[0])]))
            assert max(differences) < 1e-3
            validation["experiments_32"][name] = {"epochs": len(rows), "best_epoch": best,
                "selected_filters": expected_choice, "test_images": test["test_images"],
                "maximum_iou_difference_after_reload": max(differences)}
    external_path = directory / "carla_external_metrics.json"
    if all(path.exists() for path in comparison_files) and not external_path.exists():
        raise AssertionError("Независимая оценка CARLA ещё не выполнена")
    if external_path.exists():
        from external_carla import external_scores, verify_manifest
        manifest = verify_manifest()
        external = json.loads(external_path.read_text())
        selected = json.loads((directory / "carla_comparison.json").read_text())["selected_filters"]
        assert external["images"] == manifest["images"] == 223
        assert external["selection_from_validation"] == selected
        assert external["class_ids"] == list(range(1, 13)) and external["ignore_label"] == 0
        for width in ["8", "32"]:
            metrics = external["models"][width]
            recalculated = external_scores(metrics["confusion_matrix"])
            for key in ["accuracy", "mean_iou", "evaluated_pixels"]:
                assert abs(metrics[key] - recalculated[key]) < 1e-8
        validation["independent_carla"] = {"images": 223, "selected_filters": selected,
            "accuracy": external["models"][selected]["accuracy"],
            "mean_iou": external["models"][selected]["mean_iou"]}
    result = subprocess.run([sys.executable, "-m", "unittest", "-v",
                             "test_lab2.py", "test_real_data.py"],
                            capture_output=True, text=True)
    Path("results/tests.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    validation["tests_exit_code"] = result.returncode
    Path("results/validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    verify()
