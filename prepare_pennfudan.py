"""Скачивание и подготовка наборов CARLA и Penn-Fudan."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

CARLA_URL = "https://github.com/Rhythm-Adhikari1/U-Net.git"
CARLA_COMMIT = "072afa362ab567ceb8c151b18b00e05ac632bf26"
PENN_URL = "https://github.com/swallan/PennFudanPed.git"
PENN_COMMIT = "ec1d4583fb436b14e2062587c8b28a5018668a5e"
PARTS = ("train", "val", "test")


def download_source(url, commit, root, folder=None):
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "--depth", "1",
                    "--filter=blob:none", "origin", commit], check=True)
    if folder:
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", folder], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    return root / folder if folder else root


def verify_source(source, commit):
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != commit:
        raise ValueError(f"Нужна версия набора {commit}: {source}")
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                      "--untracked-files=no", "--", "."], text=True)
    if status.strip():
        raise ValueError(f"Исходные файлы набора изменены: {source}")


def find_pairs(images_dir, masks_dir):
    images = {p.stem: p for p in Path(images_dir).glob("*.png")}
    masks = {p.stem: p for p in Path(masks_dir).glob("*.png")}
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Проверьте соответствие изображений и масок: {images_dir}")
    return [(images[name], masks[name]) for name in sorted(images)]


def carla_split(pairs):
    groups = []
    for pair in sorted(pairs, key=lambda pair: int(pair[0].stem)):
        if not groups or int(pair[0].stem) - int(groups[-1][-1][0].stem) > 100:
            groups.append([])
        groups[-1].append(pair)
    order = np.random.default_rng(42).permutation(len(groups))
    groups = [groups[i] for i in order]

    def take_groups(available, target):
        sums = {0: []}
        for i, group in enumerate(available):
            for amount, selected in list(sums.items()):
                total = amount + len(group)
                if total <= target and total not in sums:
                    sums[total] = selected + [i]
            if target in sums:
                break
        if target not in sums:
            raise ValueError("Не удалось разделить группы кадров")
        chosen = set(sums[target])
        return ([pair for i in chosen for pair in available[i]],
                [group for i, group in enumerate(available) if i not in chosen])

    val, groups = take_groups(groups, len(pairs) // 10)
    test, groups = take_groups(groups, len(pairs) // 10)
    return {"train": [pair for group in groups for pair in group], "val": val, "test": test}


def random_split(pairs):
    order = np.random.default_rng(42).permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    n_train, n_val = int(len(pairs) * 0.8), int(len(pairs) * 0.1)
    return {"train": pairs[:n_train], "val": pairs[n_train:n_train+n_val],
            "test": pairs[n_train+n_val:]}


def organize_split(split, root):
    result = {}
    for part, pairs in split.items():
        images_dir, masks_dir = Path(root) / "images" / part, Path(root) / "masks" / part
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for image, mask in pairs:
            for source, target in ((image, images_dir / image.name), (mask, masks_dir / mask.name)):
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f"Файл уже существует и отличается: {target}")
                if not target.exists():
                    shutil.copy2(source, target)
        result[part] = find_pairs(images_dir, masks_dir)
        if len(result[part]) != len(pairs):
            raise ValueError(f"В папке {images_dir} есть лишние файлы")
    return result


def prepare_carla(source=None):
    root = Path("data/carla")
    if source is None and all((root / kind / part).is_dir() for kind in ("images", "masks") for part in PARTS):
        result = {part: find_pairs(root / "images" / part, root / "masks" / part) for part in PARTS}
        if [len(result[part]) for part in PARTS] != [848, 106, 106]:
            raise ValueError("Неверное количество изображений CARLA")
        return result
    if source is None:
        with tempfile.TemporaryDirectory(prefix="carla-") as folder:
            source = download_source(CARLA_URL, CARLA_COMMIT, Path(folder) / "source", "data")
            return prepare_carla(source)
    source = Path(source)
    verify_source(source, CARLA_COMMIT)
    pairs = find_pairs(source / "CameraRGB", source / "CameraMask")
    if len(pairs) != 1060:
        raise ValueError(f"Ожидалось 1060 пар CARLA, найдено {len(pairs)}")
    return organize_split(carla_split(pairs), root)


def prepare_penn(source):
    source = Path(source)
    verify_source(source, PENN_COMMIT)
    images = sorted((source / "PNGImages").glob("*.png"))
    if len(images) != 170:
        raise ValueError(f"Ожидалось 170 изображений Penn-Fudan, найдено {len(images)}")
    with tempfile.TemporaryDirectory(prefix="penn-masks-") as folder:
        pairs = []
        for image in images:
            source_mask = source / "PedMasks" / (image.stem + "_mask.png")
            labels = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
            original = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if labels is None or original is None or labels.ndim != 2 or labels.shape != original.shape[:2]:
                raise ValueError(f"Ошибка изображения или маски: {image.name}")
            mask = Path(folder) / image.name
            if not cv2.imwrite(str(mask), (labels > 0).astype(np.uint8)):
                raise ValueError(f"Не удалось сохранить {mask}")
            pairs.append((image, mask))
        return organize_split(random_split(pairs), "data/pennfudan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-source", type=Path, help="Путь к папке data из копии CARLA")
    parser.add_argument("--penn-source", type=Path, help="Путь к копии Penn-Fudan")
    args = parser.parse_args()
    carla = prepare_carla(args.carla_source)
    if args.penn_source:
        penn = prepare_penn(args.penn_source)
    else:
        with tempfile.TemporaryDirectory(prefix="penn-source-") as folder:
            source = download_source(PENN_URL, PENN_COMMIT, Path(folder) / "source")
            penn = prepare_penn(source)
    for name, split in (("CARLA", carla), ("Penn-Fudan", penn)):
        print(name, {part: len(pairs) for part, pairs in split.items()})


if __name__ == "__main__":
    main()
