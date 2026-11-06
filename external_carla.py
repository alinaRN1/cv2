"""Независимая оценка на опубликованной контрольной части другого набора CARLA."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cv2-matplotlib")
if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import tensorflow as tf

from lab2 import find_pairs, load_arrays, make_dataset, show_predictions
from prepare_data import CARLA_CLASSES, verify_source_revision

SOURCE = "https://github.com/cpow-89/carla_image_segmentation_dataset.git"
COMMIT = "aa0eb00150f3fda6f7950f097baa145890717979"
ROOT = Path("data/carla_external")
MANIFEST = Path("data/carla_external_manifest.json")
OUTPUT = Path("results/experiment_32")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest["files"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Контрольная сумма не совпала: {name}")
    print("Независимые данные: проверено файлов", len(manifest["files"]))
    return manifest


def prepare_from_checkout(source):
    source = Path(source)
    verify_source_revision(source / "dataset", COMMIT)
    names = (source / "dataset/valid.txt").read_text().splitlines()
    if len(names) != 223 or len(set(names)) != 223:
        raise ValueError("Неожиданный состав опубликованной контрольной части")
    if not all(re.fullmatch(r"run\d+_fr_\d+\.png", name) for name in names):
        raise ValueError("Некорректное имя файла в списке данных")
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "masks").mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "commit": COMMIT, "images": len(names),
                "class_ids": list(range(1, 13)), "ignore_label": 0, "files": {}}
    original = json.loads(Path("data/manifest.json").read_text())
    previous_images = {digest for path, digest in original["files"].items()
                       if "/images/" in path}
    for name in names:
        for source_part, target_part in [("images", "images"), ("labels", "masks")]:
            origin = source / "dataset" / source_part / name
            target = ROOT / target_part / name
            digest = sha256(origin)
            if source_part == "images" and digest in previous_images:
                raise ValueError(f"Кадр совпадает с ранее использованным набором: {name}")
            if target.exists() and sha256(target) != digest:
                raise ValueError(f"Существующий файл отличается: {target}")
            if not target.exists():
                shutil.copy2(origin, target)
            if source_part == "labels":
                mask = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.ndim != 2 or mask.max() > 12:
                    raise ValueError(f"Ожидалась маска с ID 0–12: {target}")
            manifest["files"][str(target)] = digest
    pairs = find_pairs(ROOT / "images", ROOT / "masks")
    if len(pairs) != len(names):
        raise ValueError("Лишние или отсутствующие файлы контрольного набора")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "external_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Подготовлены независимые пары:", len(pairs))


def download():
    with tempfile.TemporaryDirectory(prefix="lab2-external-") as temporary:
        repo = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--no-checkout", SOURCE, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        names = subprocess.check_output(["git", "-C", str(repo), "show",
                                         f"{COMMIT}:dataset/valid.txt"], text=True).splitlines()
        if len(names) != 223 or not all(re.fullmatch(r"run\d+_fr_\d+\.png", name) for name in names):
            raise ValueError("Некорректный список контрольных файлов")
        patterns = ["/dataset/valid.txt"] + [f"/dataset/{part}/{name}"
                    for name in names for part in ["images", "labels"]]
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone", "--stdin"],
                       input="\n".join(patterns) + "\n", text=True, check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", COMMIT], check=True)
        prepare_from_checkout(repo)


def external_scores(matrix):
    """Оцениваем классы 1–12, присутствующие в разметке; Unlabeled исключён."""
    cm = np.array(matrix, dtype=np.int64, copy=True)
    cm[0] = 0
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.full(23, np.nan), where=union > 0)
    # Один и тот же набор истинных классов используется для всех моделей.
    evaluated = [i for i in range(1, 13) if cm.sum(axis=1)[i] > 0]
    return {"accuracy": float(intersection.sum() / cm.sum()),
            "mean_iou": float(np.mean(iou[evaluated])),
            "iou_per_class": [float(iou[i]) if i in evaluated else None for i in range(23)],
            "class_pixels": cm.sum(axis=1).tolist(),
            "evaluated_pixels": int(cm.sum()), "confusion_matrix": cm.tolist()}


def evaluate():
    verify_manifest()
    comparison = json.loads((OUTPUT / "carla_comparison.json").read_text())
    pairs = find_pairs(ROOT / "images", ROOT / "masks")
    x, y = load_arrays(pairs, (96, 128), 23)
    result = {"dataset": "External CARLA 0.9.5", "source": SOURCE, "commit": COMMIT,
              "images": len(pairs), "class_ids": list(range(1, 13)), "ignore_label": 0,
              "selection_from_validation": comparison["selected_filters"], "models": {}}
    models = {"8": Path("results/carla.keras"), "32": OUTPUT / "carla_inference.keras"}
    for width, path in models.items():
        model = tf.keras.models.load_model(path, compile=False)
        confusion = np.zeros((23, 23), dtype=np.int64)
        for images, masks in make_dataset(x, y):
            predicted = tf.argmax(model(images, training=False), axis=-1).numpy().ravel()
            labels = masks.numpy().ravel()
            valid = labels != 0
            confusion += np.bincount(labels[valid] * 23 + predicted[valid], minlength=23 * 23).reshape(23, 23)
        scores = external_scores(confusion)
        result["models"][width] = scores
        print(width, "accuracy", round(scores["accuracy"], 4), "mean IoU", round(scores["mean_iou"], 4))
        if width == comparison["selected_filters"]:
            show_predictions(model, pairs, (96, 128), 23, "experiment_32/carla_external",
                             count=3, class_names=CARLA_CLASSES, ignore_label=0)
    cm = np.array(result["models"]["8"]["confusion_matrix"])
    baseline = np.zeros_like(cm)
    baseline[:, 7] = cm.sum(axis=1)
    result["baseline_road"] = external_scores(baseline)
    (OUTPUT / "carla_external_metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Число потоков должно быть положительным")
    if args.action == "prepare":
        if MANIFEST.exists():
            verify_manifest()
        elif args.source:
            prepare_from_checkout(args.source)
        else:
            download()
    else:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(2, args.threads))
        evaluate()
