"""Проверки архитектуры из задания и подготовки данных."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from lab2 import (conv_block, upsampling_block, unet_model, find_pairs,
                  split_pairs, read_pair, create_mask, label_colors,
                  scores_from_confusion, make_dataset, compile_model, worst_pairs)
from experiment_32 import WeightedCrossEntropy, weights_from_training_masks
from external_carla import external_scores
from update_work import mean_iou_present


class Lab2Tests(unittest.TestCase):
    def test_encoder(self):
        inputs = tf.keras.layers.Input((96, 128, 3))
        next_layer, skip = conv_block(inputs, 32)
        model = tf.keras.Model(inputs, [next_layer, skip])
        self.assertEqual(tuple(next_layer.shape), (None, 48, 64, 32))
        self.assertEqual(tuple(skip.shape), (None, 96, 128, 32))
        self.assertEqual([layer.count_params() for layer in model.layers], [0, 896, 9248, 0])
        next_layer, skip = conv_block(inputs, 4, 0.3, False)
        self.assertIs(next_layer, skip)
        self.assertEqual(tf.keras.Model(inputs, skip).layers[-1].rate, 0.3)

    def test_decoder(self):
        small = tf.keras.layers.Input((12, 16, 256))
        skip = tf.keras.layers.Input((24, 32, 128))
        output = upsampling_block(small, skip, 32)
        model = tf.keras.Model([small, skip], output)
        self.assertEqual(tuple(output.shape), (None, 24, 32, 32))
        counts = [layer.count_params() for layer in model.layers]
        for expected in [73760, 46112, 9248]:
            self.assertIn(expected, counts)

    def test_full_unet(self):
        model = unet_model()
        self.assertEqual(model.output_shape, (None, 96, 128, 23))
        self.assertEqual(sum(isinstance(x, tf.keras.layers.MaxPooling2D)
                             for x in model.layers), 4)
        self.assertEqual(sum(isinstance(x, tf.keras.layers.Conv2DTranspose)
                             for x in model.layers), 4)
        self.assertEqual(model.layers[-1].activation.__name__, "linear")

    def test_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            mask = Path(folder) / "mask.png"
            cv2.imwrite(str(image), np.full((8, 10, 3), 128, np.uint8))
            labels = np.zeros((8, 10, 3), np.uint8)
            labels[:, 5:, 2] = 22  # Красный канал в BGR.
            cv2.imwrite(str(mask), labels)
            x, y = read_pair(image, mask, (96, 128), 23)
            self.assertEqual(x.shape, (96, 128, 3))
            np.testing.assert_allclose(x, 128 / 255, atol=1e-7)
            self.assertEqual(set(np.unique(y)), {0, 22})
            self.assertEqual(y.dtype, np.int32)
            with self.assertRaises(ValueError):
                read_pair(image, mask, (96, 128), 4)

    def test_pairs_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for i in reversed(range(30)):
                (root / "images" / f"{i:02}.png").touch()
                (root / "masks" / f"{i:02}.png").touch()
            pairs = find_pairs(root / "images", root / "masks")
            self.assertTrue(all(x.stem == y.stem for x, y in pairs))
            split = split_pairs(pairs)
            self.assertEqual([len(v) for v in split.values()], [24, 3, 3])
            self.assertEqual(split, split_pairs(pairs))
            combined = [pair for group in split.values() for pair in group]
            self.assertEqual(len(set(combined)), 30)
            (root / "masks" / "00.png").unlink()
            with self.assertRaises(ValueError):
                find_pairs(root / "images", root / "masks")

    def test_palette_has_23_distinct_colors(self):
        colors = label_colors(23)
        self.assertEqual(colors.shape, (23, 3))
        self.assertEqual(len(np.unique(colors, axis=0)), 23)

    def test_iou_and_void(self):
        matrix = np.array([[8, 2, 0], [1, 3, 1], [9, 9, 9]])
        scores = scores_from_confusion(matrix, ignore_label=2)
        self.assertAlmostEqual(scores["accuracy"], 11 / 15)
        self.assertAlmostEqual(scores["mean_iou"], (8 / 11 + 3 / 7) / 2)
        self.assertIsNone(scores["iou_per_class"][2])
        self.assertEqual(scores["evaluated_pixels"], 15)
        self.assertEqual(matrix[2, 2], 9)  # Исходную матрицу не меняем.
        absent = scores_from_confusion([[5, 0], [0, 0]])
        self.assertEqual(absent["mean_iou"], 1)
        self.assertIsNone(absent["iou_per_class"][1])

    def test_void_is_ignored_in_loss_and_accuracy(self):
        inputs = tf.keras.layers.Input((16, 16, 3))
        outputs = tf.keras.layers.Conv2D(12, 1, kernel_initializer="zeros",
                                         bias_initializer="zeros")(inputs)
        model = tf.keras.Model(inputs, outputs)
        compile_model(model, ignore_label=11)
        x = np.zeros((1, 16, 16, 3), np.float32)
        y = np.zeros((1, 16, 16, 1), np.int32)
        y[:, :8] = 11
        scores = model.evaluate(make_dataset(x, y, ignore_label=11),
                                return_dict=True, verbose=0)
        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["loss"], np.log(12), places=5)

    def test_worst_examples_do_not_repeat_adjacent_frames(self):
        pairs = [(Path(f"{frame:06}.png"), Path(f"mask/{frame:06}.png"))
                 for frame in [0, 1, 200]]
        scores = {"per_image": [{"image": str(pair[0]), "mean_iou": i / 10}
                                for i, pair in enumerate(pairs)]}
        selected = worst_pairs(scores, pairs)
        self.assertEqual([pair[0].stem for pair in selected], ["000000", "000200"])

    def test_class_weights_are_bounded_and_ignore_void(self):
        masks = np.array([0] * 80 + [1] * 10 + [2] * 10, np.int32)
        counts, weights = weights_from_training_masks(masks, 4, ignore_label=2)
        self.assertEqual(counts.tolist(), [80, 10, 0, 0])
        self.assertGreater(weights[1], weights[0])
        self.assertEqual(weights[2], 0)
        self.assertEqual(weights[3], 0)
        self.assertAlmostEqual(float(np.dot(weights, counts / counts.sum())), 1.0)

    def test_weighted_loss_serializes_and_void_prediction_counts_as_error(self):
        loss = WeightedCrossEntropy([1.0, 2.0, 0.0])
        restored = tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
        self.assertEqual(restored.class_weights, [1.0, 2.0, 0.0])
        labels = tf.constant([[[[0], [1], [2]]]])
        logits = tf.constant([[[[0., 0., 9.], [0., 9., 0.], [0., 0., 9.]]]])
        metric = tf.keras.metrics.IoU(num_classes=3, target_class_ids=[0, 1],
                                      sparse_y_pred=False, ignore_class=2)
        metric.update_state(labels, logits)
        self.assertAlmostEqual(float(metric.result()), 0.5, places=5)

    def test_external_iou_ignores_unlabeled_and_penalizes_other_predictions(self):
        matrix = np.zeros((23, 23), np.int64)
        matrix[0, 1] = 10  # Не размечено, исключается целиком.
        matrix[1, 1], matrix[1, 2], matrix[1, 13] = 1, 1, 1
        matrix[2, 2] = 2
        matrix[2, 4] = 1  # Класс 4 отсутствует в разметке, но был предсказан.
        scores = external_scores(matrix)
        self.assertEqual(scores["evaluated_pixels"], 6)
        self.assertAlmostEqual(scores["accuracy"], 0.5)
        self.assertAlmostEqual(scores["mean_iou"], (1 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(scores["iou_per_class"][1], 1 / 3)
        self.assertAlmostEqual(scores["iou_per_class"][2], 1 / 2)
        self.assertIsNone(scores["iou_per_class"][4])
        self.assertIsNone(scores["iou_per_class"][13])

    def test_mean_iou_uses_only_ground_truth_present_classes(self):
        self.assertAlmostEqual(mean_iou_present([0.9, 0.2, 0.0], [10, 5, 0]), 0.55)
        with self.assertRaises(ValueError):
            mean_iou_present([None, 0.2], [1, 1])
        with self.assertRaises(ValueError):
            mean_iou_present([None, None], [0, 0])

    def test_prediction_and_training(self):
        tf.keras.utils.set_random_seed(42)
        model = unet_model((32, 32, 3), n_filters=2, n_classes=4)
        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True))
        x = np.random.default_rng(42).random((2, 32, 32, 3)).astype(np.float32)
        y = np.zeros((2, 32, 32, 1), np.int32)
        self.assertTrue(np.isfinite(model.train_on_batch(x, y)))
        predictions = model(x, training=False)
        self.assertEqual(tuple(create_mask(predictions).shape), (32, 32, 1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(restored(x, training=False), predictions, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
