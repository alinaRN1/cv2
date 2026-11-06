"""Дополнительный эксперимент: U-Net из задания и учёт дисбаланса классов."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cv2-matplotlib")

import numpy as np
import tensorflow as tf

from lab2 import load_arrays, make_dataset, unet_model
from prepare_data import load_split, check_manifest, CARLA_CLASSES, CAMVID_CLASSES


@tf.keras.utils.register_keras_serializable(package="Lab2")
class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, name="weighted_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = [float(value) for value in class_weights]

    def call(self, y_true, y_pred):
        labels = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=tf.cast(y_pred, tf.float32))
        return loss * tf.gather(tf.constant(self.class_weights, tf.float32), labels)

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights}


def weights_from_training_masks(masks, n_classes, ignore_label=None):
    counts = np.bincount(masks.ravel(), minlength=n_classes).astype(np.float64)
    if ignore_label is not None:
        counts[ignore_label] = 0
    frequencies = counts / counts.sum()
    # Ограничение снижает влияние очень редких классов и ошибочных пикселей.
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = 1 / np.sqrt(np.maximum(frequencies[present], 0.002))
    weights /= np.dot(weights, frequencies)
    return counts, weights


def run(name, epochs=40, threads=4):
    classes = CARLA_CLASSES if name == "carla" else CAMVID_CLASSES
    ignore = None if name == "carla" else 11
    check_manifest()
    split = load_split(name)
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))

    print(f"Загрузка {name}: train={len(split['train'])}, val={len(split['val'])}", flush=True)
    train_x, train_y = load_arrays(split["train"], (96, 128), len(classes))
    val_x, val_y = load_arrays(split["val"], (96, 128), len(classes))
    counts, weights = weights_from_training_masks(train_y, len(classes), ignore)
    output = Path("results/experiment_32")
    output.mkdir(exist_ok=True, parents=True)
    config = {"dataset": name, "epochs": epochs, "n_filters": 32, "seed": 42,
              "size": [96, 128], "n_classes": len(classes), "ignore_label": ignore,
              "cpu_threads": threads,
              "class_counts_train": counts.tolist(), "class_weights": weights.tolist(),
              "selection": "maximum validation mean IoU", "tensorflow": tf.__version__}
    (output / f"{name}_config.json").write_text(json.dumps(config, indent=2))
    print("Веса классов:", dict(zip(classes, np.round(weights, 3))), flush=True)

    model = unet_model((96, 128, 3), 32, len(classes))
    model.compile(
        optimizer="adam", loss=WeightedCrossEntropy(weights),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                          tf.keras.metrics.IoU(num_classes=len(classes),
                              target_class_ids=[i for i in range(len(classes)) if i != ignore],
                              sparse_y_pred=False, ignore_class=ignore,
                              dtype="float32", name="mean_iou")])
    print("Параметров:", model.count_params(), flush=True)
    model.fit(
        make_dataset(train_x, train_y, training=True, ignore_label=ignore),
        validation_data=make_dataset(val_x, val_y, ignore_label=ignore),
        epochs=epochs, verbose=2, shuffle=False,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            output / f"{name}_best.keras", monitor="val_mean_iou", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(output / f"{name}_history.csv")])
    print(f"Обучение {name} завершено", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["carla", "camvid"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Число потоков должно быть положительным")
    run(args.dataset, args.epochs, args.threads)
