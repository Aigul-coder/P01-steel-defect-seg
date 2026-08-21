# Скрипт к интервью — P01 сегментация дефектов стали

## Питч на 30 секунд

Сделала демо сегментации дефектов в стиле Severstal: multi-label UNet, явная работа с **дисбалансом классов** (focal + веса), per-class Dice, FastAPI с оверлеем масок и latency ONNX на CPU. Hire signal не в том, что «умею вызвать SMP», а в сравнении **baseline vs improved** с честной историей про imbalance и holdout.

## Типичные вопросы → ответы

### Почему multi-label (4 канала), а не softmax?

В разметке Severstal на одном листе могут быть **несколько классов дефектов сразу**. Softmax заставляет классы быть взаимоисключающими; независимые sigmoid-головы соответствуют схеме разметки и CSV с RLE по каждому ClassId.

### Как боролась с дисбалансом?

Большинство пикселей — фон; класс 2 обычно редкий. Baseline: Dice + BCE с равными весами. Improved: **focal BCE (γ=2)** и **повышенные pos weights** на редких классах, Dice оставляю за overlap регионов. Смотрю **per-class Dice**, а не только среднее.

### Валидация / утечки?

Сплит по **ImageId**, стратификация по наличию дефекта. **19 blind holdout** ImageId (`data/holdout_ids.txt`) исключены и из train, и из val. Один и тот же seed в YAML для train / eval / holdout.

### Зачем два числа Dice?

Обычный per-class Dice усредняет по всем картинкам: пустой GT + пустой pred → Dice = 1.0. Честнее **`per_class_dice_present_only`** из `eval_holdout.py`. На holdout: class 3 present-only у improved **0.20** против baseline **0.05**.

### Почему UNet-ResNet34?

Сильный промышленный baseline: ImageNet-энкодер, достаточно лёгкий для демо 256², простой экспорт в ONNX. FPN — переключатель в конфиге для короткой абляции.

### Какую метрику показываешь?

Таблицу на val + таблицу **blind holdout present-only**. Один mean Dice маскирует collapse головы (например class 4 на val ~0.94 Dice, а на holdout recall class 4 = 0).

### ONNX / сервис?

Обучение в PyTorch (CUDA, если есть). Экспорт ONNX; инференс Torch или ORT через `INFERENCE_BACKEND`. API отдаёт площади по классам + **PNG-оверлей** для визуального QA.

### Типичные сбои?

- **Multi-class collapse:** дефекты в основном как class 3; class 4 (жёлтый GT) на holdout почти не предсказывается  
- Тонкие царапины vs текстура проката → ложные срабатывания  
- Resize/pad до 256² меняет геометрию тонких дефектов  

### Что не дотягивает до продакшена?

Нет приватных данных комбината, TensorRT не обязателен, нет фарма Kaggle leaderboard. Дальше: группировка на уровне листа, сильнее аугментации, калибровка порогов по классам.

## Команды демо (запомнить)

```bash
python scripts/download_data.py --source synthetic --subset 64
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/improved.yaml
python scripts/export_onnx.py --config configs/improved.yaml
docker compose -f docker/docker-compose.yml up --build
curl -s http://127.0.0.1:8000/health
```
