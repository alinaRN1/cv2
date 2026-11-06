"""Скачивание двух наборов и сохранение проверяемых разбиений."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lab2 import find_pairs, organize_split

SOURCES = {
    "carla": {"url": "https://github.com/Rhythm-Adhikari1/U-Net.git",
              "commit": "072afa362ab567ceb8c151b18b00e05ac632bf26", "folder": "data"},
    "camvid": {"url": "https://github.com/alexgkendall/SegNet-Tutorial.git",
               "commit": "bb68b64e72cb004b3ed927fee0f5624dd8097498", "folder": "CamVid"},
}

CARLA_CLASSES = ["Unlabeled", "Building", "Fence", "Other", "Pedestrian", "Pole",
                 "RoadLine", "Road", "Sidewalk", "Vegetation", "Vehicle", "Wall",
                 "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GuardRail",
                 "TrafficLight", "Static", "Dynamic", "Water", "Terrain"]
CAMVID_CLASSES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol",
                  "Fence", "Car", "Pedestrian", "Bicyclist", "Void"]


def verify_source_revision(folder, expected):
    """Убеждаемся, что копируем отслеживаемые файлы из нужного коммита."""
    folder = Path(folder).resolve()
    result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != expected:
        raise ValueError(f"Источник {folder} должен быть checkout коммита {expected}")
    root = Path(subprocess.check_output(["git", "-C", str(folder), "rev-parse",
                                         "--show-toplevel"], text=True).strip()).resolve()
    relative = str(folder.relative_to(root))
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                      "--untracked-files=no", "--", relative], text=True)
    if status.strip():
        raise ValueError(f"Отслеживаемые файлы источника {folder} изменены")


def carla_split(pairs):
    """Соседние номера кадров не должны попасть в разные части."""
    groups = []
    for pair in sorted(pairs, key=lambda p: int(p[0].stem)):
        if not groups or int(pair[0].stem) - int(groups[-1][-1][0].stem) > 100:
            groups.append([])
        groups[-1].append(pair)
    order = np.random.default_rng(42).permutation(len(groups))
    groups = [groups[i] for i in order]

    # Находим целые группы суммарно на 10% кадров: так сохраняется точное 80/10/10.
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
            raise ValueError("Не удалось получить 10% без разделения соседних кадров")
        chosen = set(sums[target])
        return ([pair for i in chosen for pair in available[i]],
                [group for i, group in enumerate(available) if i not in chosen])

    val, groups = take_groups(groups, len(pairs) // 10)
    test, groups = take_groups(groups, len(pairs) // 10)
    return {"train": [pair for group in groups for pair in group], "val": val, "test": test}


def prepare_from_sources(carla_root, camvid_root):
    verify_source_revision(carla_root, SOURCES["carla"]["commit"])
    verify_source_revision(camvid_root, SOURCES["camvid"]["commit"])
    carla_pairs = find_pairs(Path(carla_root) / "CameraRGB", Path(carla_root) / "CameraMask")
    if len(carla_pairs) != 1060:
        raise ValueError("Ожидалось 1060 пар исходного учебного набора CARLA")
    carla = organize_split(carla_split(carla_pairs), "data/carla")
    # CamVid имеет опубликованное разбиение: не перемешиваем его заново.
    camvid = {part: find_pairs(Path(camvid_root) / part, Path(camvid_root) / (part + "annot"))
              for part in ["train", "val", "test"]}
    if [len(camvid[part]) for part in camvid] != [367, 101, 233]:
        raise ValueError("Состав CamVid не соответствует опубликованному разбиению")
    camvid = organize_split(camvid, "data/camvid")
    manifest = {"sources": SOURCES,
                "carla_split": "80/10/10; nearby frame IDs grouped with max gap 100; seed 42",
                "camvid_split": "official 367/101/233; void label 11 ignored",
                "files": {}}
    for name, split in [("carla", carla), ("camvid", camvid)]:
        for part, pairs in split.items():
            for image, mask in pairs:
                for path in [image, mask]:
                    manifest["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(name, {key: len(value) for key, value in split.items()})
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/data_manifest.json").write_text(json.dumps(manifest, indent=2))
    return carla, camvid


def load_split(name):
    return {part: find_pairs(Path("data") / name / "images" / part,
                            Path("data") / name / "masks" / part)
            for part in ["train", "val", "test"]}


def check_manifest():
    manifest = json.loads(Path("data/manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Контрольная сумма не совпала: {filename}")
    print("Проверены SHA-256 для", len(manifest["files"]), "файлов")


def download_source(name, target):
    source = SOURCES[name]
    def git(*args):
        subprocess.run(["git", *args], check=True)
    git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", source["url"], str(target))
    git("-C", str(target), "fetch", "--depth", "1", "origin", source["commit"])
    git("-C", str(target), "sparse-checkout", "set", source["folder"])
    git("-C", str(target), "checkout", "--detach", source["commit"])
    return target / source["folder"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-source", type=Path)
    parser.add_argument("--camvid-source", type=Path)
    args = parser.parse_args()
    if Path("data/manifest.json").exists():
        check_manifest()
    elif args.carla_source and args.camvid_source:
        prepare_from_sources(args.carla_source, args.camvid_source)
    else:
        with tempfile.TemporaryDirectory(prefix="lab2-data-") as folder:
            root = Path(folder)
            carla = download_source("carla", root / "carla")
            camvid = download_source("camvid", root / "camvid")
            prepare_from_sources(carla, camvid)
