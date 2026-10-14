import os
import random
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============ 1. НАСТРОЙКИ ============
WEIGHTS   = 'runs/detect/runs/finetune/construction_equipment_subset/weights/best.pt'  # ← ваш чекпоинт 36 эпох
DATA_YAML = 'data.yaml'
IMGSZ     = 640
CONF      = 0.25          # порог уверенности
IOU       = 0.5           # порог IoU для NMS
DEVICE    = 'cpu'             # 0 или 'cpu'
N_SHOW    = 9             # сколько изображений показать в сетке
SEED      = 42

random.seed(SEED)


# ============ 2. ПРОВЕРКА ============
weights_path = Path(WEIGHTS)
if not weights_path.exists():
    # Попробуем best.pt, если last.pt нет
    alt = weights_path.parent / 'best.pt'
    if alt.exists():
        weights_path = alt
        print(f"⚠️ last.pt не найден, использую {alt}")
    else:
        raise FileNotFoundError(f"Не найдены веса: {WEIGHTS}")

print(f"📦 Модель: {weights_path}")
model = YOLO(str(weights_path))
assert model.task == 'detect', f"Модель не для детекции: task={model.task}"


# ============ 3. МЕТРИКИ НА VAL ============
print(f"\n📊 Валидация на val-сплите...")
metrics = model.val(
    data=DATA_YAML,
    split='val',
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    device=DEVICE,
    plots=True,
    save_json=False,
)

print(f"\n=== Итоговые метрики ===")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")

# Per-class метрики
print(f"\n=== По классам ===")
names = model.names
try:
    for i, (p, r, ap50, ap) in enumerate(zip(
        metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
    )):
        print(f"  {i:2d} {names[i]:<20s} P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}")
except Exception as e:
    print(f"  (не удалось вывести per-class: {e})")


# ============ 4. ЗАГРУЗКА ПУТЕЙ К ВАЛИДАЦИОННЫМ ИЗОБРАЖЕНИЯМ ============
cfg = yaml.safe_load(Path(DATA_YAML).read_text())
root = Path(cfg.get('path', Path(DATA_YAML).parent)).resolve()
val_dir = (root / cfg['val']).resolve()

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
val_images = [p for p in val_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
print(f"\n🖼️  Найдено val-изображений: {len(val_images)}")

sample = random.sample(val_images, min(N_SHOW, len(val_images)))


# ============ 5. ПРЕДСКАЗАНИЯ ============
print(f"\n🔮 Предсказания для {len(sample)} изображений...")
results = model.predict(
    source=[str(p) for p in sample],
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    device=DEVICE,
    verbose=False,
)


# ============ 6. ВИЗУАЛИЗАЦИЯ ============
def plot_results(results, n_cols=3):
    n = len(results)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, results):
        # Встроенный метод: рисует рамки и подписи
        annotated = r.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
            font_size=12,
        )
        # BGR → RGB для matplotlib
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(r.path).name, fontsize=10)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    return fig


fig = plot_results(results, n_cols=3)
out_path = Path(WEIGHTS).parent / 'val_visualization.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\n💾 Визуализация сохранена: {out_path.resolve()}")


# ============ 7. ПОКАЗ НА ЭКРАНЕ ============
plt.show()   # интерактивное окно


# ============ 8. ОТДЕЛЬНЫЙ ПРОСМОТР КАЖДОГО ИЗОБРАЖЕНИЯ (опционально) ============
# Раскомментируйте, если хотите смотреть по одному в большом размере с навигацией.

# for i, r in enumerate(results):
#     annotated = r.plot(conf=True, line_width=2)
#     win = f"Prediction {i+1}/{len(results)} — {Path(r.path).name}"
#     cv2.imshow(win, annotated)
#     key = cv2.waitKey(0) & 0xFF
#     cv2.destroyWindow(win)
#     if key == 27:  # ESC — выход
#         break
# cv2.destroyAllWindows()


# ============ 9. СВОДКА ПО ПРЕДСКАЗАНИЯМ ============
print(f"\n=== Детекций на изображение ===")
for r in results:
    n_boxes = 0 if r.boxes is None else len(r.boxes)
    cls_names = []
    if n_boxes:
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        cls_names = [names[c] for c in cls_ids]
    print(f"  {Path(r.path).name:<40s} boxes={n_boxes:3d}  {cls_names}")