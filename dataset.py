import kagglehub

# Download latest version
path = kagglehub.dataset_download("xyzyxzzxy/construction-equipment")

print("Path to dataset files:", path)

import os
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
# ============ НАСТРОЙКИ ============
KEEP_RATIO = 0.3          # Доля сохраняемых изображений (0.3 = 30%)
SEED = 42
OUTPUT_DIR = Path("./construction_equipment_subset")

# ============ СКАЧИВАНИЕ ============
dataset_path = Path(kagglehub.dataset_download("xyzyxzzxy/construction-equipment"))
print(f"📦 Оригинал: {dataset_path}")

# ============ АВТООПРЕДЕЛЕНИЕ СПЛИТОВ ============
splits = []
for candidate in ['train', 'val', 'valid', 'test']:
    p = dataset_path / candidate
    if p.exists() and (p / 'images').exists() and (p / 'labels').exists():
        splits.append(candidate)

if not splits:
    if (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
        splits = ['.']
    else:
        raise RuntimeError("Не найдена структура images/labels")

print(f"Найдены сплиты: {splits}")

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
def parse_label(label_path: Path):
    """Возвращает список class_id из файла разметки."""
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(int(float(parts[0])))
    return classes


def find_image(label_path: Path, labels_root: Path, images_root: Path):
    """Ищет изображение, зеркально повторяя относительный путь label'а."""
    rel = label_path.relative_to(labels_root)
    for ext in IMAGE_EXTS:
        cand = images_root / rel.parent / f"{rel.stem}{ext}"
        if cand.exists():
            return cand
    return None


def stratified_sample(records, keep_ratio, class_counter, rng):
    """Стратификация по самому редкому классу в изображении."""
    # Определяем страту для каждого изображения
    for r in records:
        if r['class_set']:
            r['strata'] = min(r['class_set'], key=lambda c: class_counter[c])
        else:
            r['strata'] = 'background'  # изображения без объектов

    groups = defaultdict(list)
    for r in records:
        groups[r['strata']].append(r)

    selected = []
    for key, items in groups.items():
        n_keep = max(1, round(len(items) * keep_ratio))  # минимум 1 из страты
        n_keep = min(n_keep, len(items))
        rng.shuffle(items)
        selected.extend(items[:n_keep])
    return selected, groups


# ============ ОСНОВНОЙ ЦИКЛ ============
rng = random.Random(SEED)

for split in splits:
    if split == '.':
        labels_dir = dataset_path / 'labels'
        images_dir = dataset_path / 'images'
        out_labels = OUTPUT_DIR / 'labels'
        out_images = OUTPUT_DIR / 'images'
    else:
        labels_dir = dataset_path / split / 'labels'
        images_dir = dataset_path / split / 'images'
        out_labels = OUTPUT_DIR / split / 'labels'
        out_images = OUTPUT_DIR / split / 'images'

    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    label_files = list(labels_dir.rglob('*.txt'))
    print(f"\n[{split}] Найдено .txt разметок: {len(label_files)}")

    # ---- Парсинг ----
    class_counter = Counter()
    records = []
    for lbl in label_files:
        img = find_image(lbl, labels_dir, images_dir)
        if img is None:
            continue
        classes = parse_label(lbl)
        for c in classes:
            class_counter[c] += 1
        records.append({
            'label': lbl,
            'image': img,
            'classes': classes,
            'class_set': set(classes),
        })

    print(f"  Частота классов (instances): {dict(sorted(class_counter.items()))}")

    # ---- Стратифицированный сэмпл ----
    selected, groups = stratified_sample(records, KEEP_RATIO, class_counter, rng)

    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        kept = len([r for r in selected if r['strata'] == key])
        print(f"  Страта '{key}': {len(items)} → {kept}")

    # ---- Копирование с сохранением структуры ----
    for r in selected:
        rel_label = r['label'].relative_to(labels_dir)
        rel_image = r['image'].relative_to(images_dir)

        (out_labels / rel_label.parent).mkdir(parents=True, exist_ok=True)
        (out_images / rel_image.parent).mkdir(parents=True, exist_ok=True)

        shutil.copy2(r['label'], out_labels / rel_label)
        shutil.copy2(r['image'], out_images / rel_image)

    print(f"[{split}] ✅ Сохранено пар: {len(selected)}")

print(f"\n🎉 Готово! Подмножество: {OUTPUT_DIR.resolve()}")


