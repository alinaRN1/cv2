"""Предсказания обученных моделей на собственных фотографиях."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import tensorflow as tf

from lab2 import label_colors
from prepare_data import CARLA_CLASSES, CAMVID_CLASSES

PHOTOS = [Path('photos/city.jpg'), Path('photos/harbor.jpg'),
          Path('photos/mountain_road.jpg'), Path('photos/road_and_stream.jpg')]


def show_own_photos(name):
    classes = CARLA_CLASSES if name == 'carla' else CAMVID_CLASSES
    paths = [Path(f'results/{name}.keras'),
             Path(f'results/experiment_32/{name}_inference.keras')]
    images = []
    for path in PHOTOS:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image, (128, 96), interpolation=cv2.INTER_AREA) / 255.0)
    images = np.asarray(images, dtype=np.float32)
    predictions = []
    for path in paths:
        model = tf.keras.models.load_model(path, compile=False)
        predictions.append(np.argmax(model(images, training=False).numpy(), axis=-1))

    colors = label_colors(len(classes))
    fig, axes = plt.subplots(len(images), 3, figsize=(10, 10), squeeze=False)
    for i, (image, photo) in enumerate(zip(images, PHOTOS)):
        for axis, content in zip(axes[i], [image, colors[predictions[0][i]], colors[predictions[1][i]]]):
            axis.imshow(content)
            axis.axis('off')
        axes[i, 0].set_title(photo.stem.replace('_', ' '))
        axes[i, 1].set_title('U-Net, 8 фильтров')
        axes[i, 2].set_title('U-Net, 32 фильтра')
    present = np.unique(np.stack(predictions))
    handles = [Patch(color=colors[i] / 255, label=f'{i}: {classes[i]}') for i in present]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8)
    fig.subplots_adjust(bottom=0.13, hspace=0.25)
    output = Path(f'results/{name}_own_photos.png')
    fig.savefig(output, dpi=140, bbox_inches='tight')
    plt.show()
    return predictions


if __name__ == '__main__':
    for dataset in ('carla', 'camvid'):
        show_own_photos(dataset)
