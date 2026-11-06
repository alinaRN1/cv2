"""Подготовка CARLA и набора Penn-Fudan для лабораторной работы."""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from lab2 import find_pairs, organize_split, split_pairs
from prepare_data import carla_split, download_source, verify_source_revision, SOURCES

PENN_URL = "https://github.com/swallan/PennFudanPed.git"
PENN_COMMIT = "ec1d4583fb436b14e2062587c8b28a5018668a5e"


def prepare_carla():
    root = Path("data/carla")
    if all((root / kind / part).is_dir() for kind in ("images", "masks")
           for part in ("train", "val", "test")):
        return {part: find_pairs(root / "images" / part, root / "masks" / part)
                for part in ("train", "val", "test")}
    with tempfile.TemporaryDirectory(prefix="carla-") as folder:
        source = download_source("carla", Path(folder) / "source")
        verify_source_revision(source, SOURCES["carla"]["commit"])
        pairs = find_pairs(source / "CameraRGB", source / "CameraMask")
        return organize_split(carla_split(pairs), root)


def prepare_penn(source):
    source = Path(source)
    verify_source_revision(source, PENN_COMMIT)
    images = sorted((source / "PNGImages").glob("*.png"))
    if len(images) != 170:
        raise ValueError(f"Ожидалось 170 изображений Penn-Fudan, найдено {len(images)}")
    with tempfile.TemporaryDirectory(prefix="penn-masks-") as folder:
        masks = Path(folder)
        pairs = []
        for image in images:
            source_mask = source / "PedMasks" / (image.stem + "_mask.png")
            labels = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
            original = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if labels is None or original is None or labels.ndim != 2 or labels.shape != original.shape[:2]:
                raise ValueError(f"Ошибка изображения или маски: {image.name}")
            mask = masks / image.name
            if not cv2.imwrite(str(mask), (labels > 0).astype(np.uint8)):
                raise ValueError(f"Не удалось сохранить {mask}")
            pairs.append((image, mask))
        return organize_split(split_pairs(pairs), "data/pennfudan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--penn-source", type=Path, help="Локальная копия Penn-Fudan")
    args = parser.parse_args()
    carla = prepare_carla()
    if args.penn_source:
        penn = prepare_penn(args.penn_source)
    else:
        with tempfile.TemporaryDirectory(prefix="penn-source-") as folder:
            repo = Path(folder) / "repo"
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", PENN_URL], check=True)
            subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "--depth", "1", "origin", PENN_COMMIT], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
            penn = prepare_penn(repo)
    for name, split in (("CARLA", carla), ("Penn-Fudan", penn)):
        print(name, {part: len(pairs) for part, pairs in split.items()})


if __name__ == "__main__":
    main()
