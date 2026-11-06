"""Функции для лабораторной № 2: данные, U-Net и оценка сегментации."""

import csv
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.patches import Patch

SEED = 42


def conv_block(inputs, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = layers.Conv2D(n_filters, 3, activation="relu", padding="same",
                         kernel_initializer="he_normal")(inputs)
    conv = layers.Conv2D(n_filters, 3, activation="relu", padding="same",
                         kernel_initializer="he_normal")(conv)
    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)
    next_layer = layers.MaxPooling2D(2)(conv) if max_pooling else conv
    return next_layer, conv


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = layers.Conv2DTranspose(n_filters, 3, strides=2,
                                padding="same")(expansive_input)
    conv = layers.concatenate([up, contractive_input], axis=3)
    for _ in range(2):
        conv = layers.Conv2D(n_filters, 3, activation="relu", padding="same",
                             kernel_initializer="he_normal")(conv)
    return conv


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = layers.Input(input_size)
    c1 = conv_block(inputs, n_filters)
    c2 = conv_block(c1[0], n_filters * 2)
    c3 = conv_block(c2[0], n_filters * 4)
    c4 = conv_block(c3[0], n_filters * 8, dropout_prob=0.3)
    c5 = conv_block(c4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5[0], c4[1], n_filters * 8)
    u7 = upsampling_block(u6, c3[1], n_filters * 4)
    u8 = upsampling_block(u7, c2[1], n_filters * 2)
    u9 = upsampling_block(u8, c1[1], n_filters)
    conv = layers.Conv2D(n_filters, 3, activation="relu", padding="same",
                         kernel_initializer="he_normal")(u9)
    # Без softmax: функция потерь принимает logits.
    outputs = layers.Conv2D(n_classes, 1, padding="same")(conv)
    return tf.keras.Model(inputs, outputs, name="unet")


def find_pairs(image_dir, mask_dir):
    """Сопоставляем по имени, а не по порядку файлов в папках."""
    images = {p.stem: p for p in Path(image_dir).glob("*.png")}
    masks = {p.stem: p for p in Path(mask_dir).glob("*.png")}
    if not images or not masks:
        raise FileNotFoundError(f"Нужны PNG-изображения в {image_dir} и маски в {mask_dir}")
    if images.keys() != masks.keys():
        raise ValueError("Имена изображений и масок не совпадают")
    return [(images[name], masks[name]) for name in sorted(images)]


def split_pairs(pairs, shuffle=True):
    if len(pairs) < 10:
        raise ValueError("Для разбиения 80/10/10 нужно хотя бы 10 изображений")
    # Для видеокадров сохраняем временной порядок, чтобы не перемешивать соседей.
    order = (np.random.default_rng(SEED).permutation(len(pairs)) if shuffle
             else np.arange(len(pairs)))
    pairs = [pairs[i] for i in order]
    n_train, n_val = int(len(pairs) * 0.8), int(len(pairs) * 0.1)
    return {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:]}


def read_pair(image_path, mask_path, size, n_classes):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError(f"Не удалось прочитать пару {image_path}, {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Размеры изображения и маски не совпадают")
    if mask.ndim == 3:
        # В CARLA метка хранится в красном канале, OpenCV читает BGR.
        red = mask[..., 2]
        for channel in (mask[..., 0], mask[..., 1]):
            if not np.all((channel == 0) | (channel == red)):
                raise ValueError("Нужна маска с ID классов, а не цветная визуализация")
        mask = red
    if mask.min() < 0 or mask.max() >= n_classes:
        raise ValueError(f"Маска {mask_path}: ID классов должны быть 0..{n_classes - 1}")
    height, width = size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image.astype(np.float32) / 255.0, mask.astype(np.int32)[..., None]


def load_arrays(pairs, size, n_classes):
    samples = [read_pair(image, mask, size, n_classes) for image, mask in pairs]
    return np.stack([x for x, _ in samples]), np.stack([y for _, y in samples])


def make_dataset(images, masks, batch_size=8, training=False, ignore_label=None):
    if ignore_label is None:
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    else:
        weights = (masks[..., 0] != ignore_label).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, masks, weights))
    if training:
        dataset = dataset.shuffle(len(images), seed=SEED)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def save_split(split, name):
    Path("results").mkdir(exist_ok=True)
    data = {key: [str(image) for image, _ in pairs] for key, pairs in split.items()}
    Path(f"results/{name}_split.json").write_text(json.dumps(data, indent=2))


def organize_split(split, root="data/shapes"):
    """Сохраняем отдельные папки train/val/test по структуре из задания."""
    result = {}
    for part, pairs in split.items():
        image_dir = Path(root) / "images" / part
        mask_dir = Path(root) / "masks" / part
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for image, mask in pairs:
            for source, target in [(image, image_dir / image.name),
                                   (mask, mask_dir / mask.name)]:
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f"Файл {target} уже существует с другими данными")
                if not target.exists():
                    shutil.copy2(source, target)
        result[part] = find_pairs(image_dir, mask_dir)
        if len(result[part]) != len(pairs):
            raise ValueError(f"В {image_dir} есть лишние файлы от другого набора")
    return result


def compile_model(model, ignore_label=None):
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, ignore_class=ignore_label),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])


def train_model(split, name, size, n_classes, epochs=40, n_filters=8, ignore_label=None,
                reuse_existing=False):
    saved = [Path(f"results/{name}{suffix}")
             for suffix in [".keras", "_run.json", "_history.csv", "_split.json"]]
    if reuse_existing and all(path.exists() for path in saved):
        run = json.loads(saved[1].read_text())
        expected = {"size": list(size), "n_classes": n_classes, "epochs": epochs,
                    "n_filters": n_filters, "ignore_label": ignore_label, "seed": SEED}
        actual_split = {part: [str(x) for x, _ in pairs] for part, pairs in split.items()}
        if any(run[key] != value for key, value in expected.items()) or actual_split != json.loads(saved[3].read_text()):
            raise ValueError("Сохранённый эксперимент имеет другие параметры; нужен новый запуск")
        rows = list(csv.DictReader(saved[2].open()))
        if len(rows) != epochs:
            raise ValueError("История обучения неполная")
        history = SimpleNamespace(epoch=list(range(epochs)), history={
            key: [float(row[key]) for row in rows] for key in rows[0] if key != "epoch"})
        model = tf.keras.models.load_model(saved[0], compile=False)
        compile_model(model, ignore_label)
        print(f"{name}: загружен выполненный эксперимент ({epochs} эпох), лучшая эпоха {run['best_epoch']}")
        return model, history
    tf.keras.utils.set_random_seed(SEED)
    train_x, train_y = load_arrays(split["train"], size, n_classes)
    val_x, val_y = load_arrays(split["val"], size, n_classes)
    model = unet_model((*size, 3), n_filters, n_classes)
    compile_model(model, ignore_label)
    Path("results").mkdir(exist_ok=True)
    save_split(split, name)
    history = model.fit(
        make_dataset(train_x, train_y, training=True, ignore_label=ignore_label),
        validation_data=make_dataset(val_x, val_y, ignore_label=ignore_label),
        epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            f"results/{name}.keras", monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(f"results/{name}_history.csv")])
    best_model = tf.keras.models.load_model(f"results/{name}.keras")
    # В итоговый файл не включаем состояние Adam — для предсказания оно не нужно.
    inference_model = tf.keras.Model.from_config(best_model.get_config())
    inference_model.set_weights(best_model.get_weights())
    inference_model.save(f"results/{name}.keras")
    Path(f"results/{name}_run.json").write_text(json.dumps({
        "epochs": epochs, "n_filters": n_filters, "size": size, "seed": SEED,
        "n_classes": n_classes, "ignore_label": ignore_label,
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "tensorflow": tf.__version__, "parameters": model.count_params(),
        "split_sizes": {key: len(value) for key, value in split.items()}}, indent=2))
    return best_model, history


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    epochs = np.arange(1, len(history.epoch) + 1)
    for axis, metric, title in zip(axes, ["loss", "accuracy"], ["Loss", "Pixel accuracy"]):
        axis.plot(epochs, history.history[metric], label="train")
        axis.plot(epochs, history.history[f"val_{metric}"], label="validation")
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{name}_history.png", dpi=140)
    plt.show()


def create_mask(pred_mask):
    return tf.argmax(pred_mask, axis=-1)[..., tf.newaxis][0]


def label_colors(n_classes):
    """Дискретная палитра: отдельный цвет для каждого ID, включая все 23 класса."""
    colors = np.zeros((n_classes, 3), np.uint8)
    for label in range(n_classes):
        value = label
        for bit in range(8):
            for channel in range(3):
                colors[label, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    return colors


def show_predictions(model, pairs, size, n_classes, name, count=3, class_names=None,
                     ignore_label=None):
    indices = np.linspace(0, len(pairs) - 1, min(count, len(pairs)), dtype=int)
    images, masks = load_arrays([pairs[i] for i in indices], size, n_classes)
    predictions = np.argmax(model(images, training=False).numpy(), axis=-1)
    # Области без разметки исключены из метрик и не показываются как ошибки.
    if ignore_label is not None:
        predictions[masks[..., 0] == ignore_label] = ignore_label
    fig, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)), squeeze=False)
    colors = label_colors(n_classes)
    for i in range(len(images)):
        axes[i, 0].imshow(images[i])
        axes[i, 1].imshow(colors[masks[i, ..., 0]])
        axes[i, 2].imshow(colors[predictions[i]])
        for axis, title in zip(axes[i], ["Image", "True mask", "Prediction"]):
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    if class_names:
        present = np.union1d(np.unique(masks), np.unique(predictions))
        handles = [Patch(color=colors[c] / 255, label=f"{c}: {class_names[c]}")
                   for c in present]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
        fig.subplots_adjust(bottom=0.14)
    Path("results").mkdir(exist_ok=True)
    fig.savefig(f"results/{name}_predictions.png", dpi=140)
    plt.show()


def scores_from_confusion(confusion, ignore_label=None):
    confusion = np.array(confusion, dtype=np.int64, copy=True)
    if ignore_label is not None:
        confusion[ignore_label] = 0  # Пиксели void не оцениваются.
    intersection = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.full(len(union), np.nan), where=union > 0)
    if ignore_label is not None:
        iou[ignore_label] = np.nan
    return {"accuracy": float(intersection.sum() / confusion.sum()),
            "mean_iou": float(np.nanmean(iou)),
            "iou_per_class": [float(v) if np.isfinite(v) else None for v in iou],
            "class_pixels": confusion.sum(axis=1).tolist(),
            "evaluated_pixels": int(confusion.sum()), "confusion_matrix": confusion.tolist()}


def evaluate_model(model, pairs, size, n_classes, name, ignore_label=None,
                   baseline_class=None, class_names=None):
    images, masks = load_arrays(pairs, size, n_classes)
    dataset = make_dataset(images, masks, ignore_label=ignore_label)
    scores = model.evaluate(dataset, verbose=0, return_dict=True)
    confusion = np.zeros((n_classes, n_classes), np.int64)
    per_image = []
    for batch in dataset:
        x, y = batch[:2]
        pred = np.argmax(model(x, training=False).numpy(), axis=-1)
        for true_mask, prediction in zip(y.numpy()[..., 0], pred):
            matrix = tf.math.confusion_matrix(
                true_mask.ravel(), prediction.ravel(), num_classes=n_classes).numpy()
            if ignore_label is not None:
                matrix[ignore_label] = 0
            confusion += matrix
            per_image.append(scores_from_confusion(matrix, ignore_label)["mean_iou"])
    scores.update(scores_from_confusion(confusion, ignore_label))
    scores["test_images"] = len(pairs)
    scores["ignore_label"] = ignore_label
    scores["class_names"] = class_names
    if baseline_class is not None:
        baseline = np.zeros_like(confusion)
        baseline[:, baseline_class] = confusion.sum(axis=1)
        scores["majority_baseline"] = scores_from_confusion(baseline, ignore_label)
        scores["majority_baseline"]["class"] = int(baseline_class)
    scores["per_image"] = [{"image": str(pair[0]), "mean_iou": score}
                           for pair, score in zip(pairs, per_image)]
    Path(f"results/{name}_metrics.json").write_text(json.dumps(scores, indent=2))
    return scores


def worst_pairs(scores, pairs, count=2):
    """Худшие кадры без нескольких соседей одной и той же серии."""
    by_path = {str(image): (image, mask) for image, mask in pairs}
    selected, frame_ids = [], []
    for item in sorted(scores["per_image"], key=lambda x: x["mean_iou"]):
        pair = by_path[item["image"]]
        match = re.match(r"(.*?)(\d+)$", pair[0].stem)
        prefix, frame = match[1], int(match[2])
        if any(prefix == p and abs(frame - n) <= 100 for p, n in frame_ids):
            continue
        selected.append(pair)
        frame_ids.append((prefix, frame))
        if len(selected) == count:
            break
    return selected


def inspect_dataset(split, name, size, class_names, ignore_label=None):
    """Проверяем каждую пару и считаем распределение классов после resize."""
    n_classes = len(class_names)
    stats = {}
    for part, pairs in split.items():
        counts = np.zeros(n_classes, np.int64)
        for image, mask in pairs:
            _, labels = read_pair(image, mask, size, n_classes)
            counts += np.bincount(labels.ravel(), minlength=n_classes)
        stats[part] = {"images": len(pairs), "class_pixels": counts.tolist()}
    counts = np.array(stats["train"]["class_pixels"])
    if ignore_label is not None:
        counts[ignore_label] = 0
    stats["majority_class"] = int(counts.argmax())
    stats["class_names"] = class_names
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}_data.json").write_text(json.dumps(stats, indent=2))
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(np.arange(n_classes), counts / counts.sum())
    ax.set_xticks(np.arange(n_classes), class_names, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Train pixel fraction")
    fig.tight_layout()
    fig.savefig(f"results/{name}_classes.png", dpi=140)
    plt.show()
    return stats
