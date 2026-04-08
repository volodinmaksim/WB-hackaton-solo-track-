# WB Hackathon Solo Track

Репозиторий с итоговым воспроизводимым решением для задачи прогноза `target_1h` по данным `route_id + timestamp`.

## Коротко о результате

- Лучшее подтверждённое решение: `LightGBM + smoothing агрегатов`
- Public leaderboard score: `0.3608341250041473`
- Финальный сабмит из хакатона: [artifacts/best_submission/submission_route_time_agg_smooth_v1.csv](./artifacts/best_submission/submission_route_time_agg_smooth_v1.csv)

## Идея решения

Задача имела жёсткое ограничение: в `test` доступны только:

- `id`
- `route_id`
- `timestamp`

Поэтому в финальной модели нельзя было использовать:

- `status_1..6`
- лаги `target_1h`

Финальная рабочая идея:

1. Извлечь честные календарные признаки из `timestamp`
2. Посчитать по `train` исторические сглаженные агрегаты:
   - по `route_id`
   - по `(route_id, hour_slot)`
   - по `(route_id, dayofweek)`
   - по `(route_id, hour_slot, dayofweek)`
   - глобально по `hour_slot`
   - глобально по `dayofweek`
3. Обучить `LightGBM`
4. Предсказать `test`

Сглаживание оказалось ключевым улучшением. Простые `mean`-агрегаты были слабее, а `count`, `median`, blend и дополнительные редкие агрегаты не дали прироста на leaderboard.

## Структура репозитория

- [run_final_solution.py](./run_final_solution.py) — основной воспроизводимый скрипт решения
- [configs/final_solution.json](./configs/final_solution.json) — конфигурация финальной версии
- [docs/report.md](./docs/report.md) — краткий отчёт по решению
- [leaderboard_notes.md](./leaderboard_notes.md) — журнал экспериментов
- [presentation/wb_hackathon_solution.js](./presentation/wb_hackathon_solution.js) — исходник презентации
- `presentation/wb_hackathon_solution.pptx` — презентация
- [artifacts/final_solution](./artifacts/final_solution) — результаты запуска воспроизводимого скрипта
- [artifacts/best_submission](./artifacts/best_submission) — лучший сабмит из хакатона

## Как воспроизвести решение

### 1. Подготовить данные

Файлы соревнования нужно положить в папку `data/` с именами:

- `data/train_solo_track.parquet`
- `data/test_solo_track.parquet`

Датасеты не закоммичены в репозиторий, потому что это артефакты соревнования.

### 2. Установить зависимости

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Запустить финальное решение

```bash
python run_final_solution.py
```

По умолчанию будут созданы:

- `artifacts/final_solution/submission.csv`
- `artifacts/final_solution/metrics.json`
- `artifacts/final_solution/feature_importance.csv`

### 4. Запуск с явными путями

```bash
python run_final_solution.py ^
  --train-path data/train_solo_track.parquet ^
  --test-path data/test_solo_track.parquet ^
  --output-dir artifacts/final_solution
```

## Локальная валидация

Схема валидации была честной:

- последние `8` глобальных timestamp уходят в validation
- все агрегаты считаются только по `fit`-части
- затем применяются к `validation`

Лучшая локальная метрика для финальной smoothing-версии:

- `score = 0.3835836125379652`

## Что не сработало

- `count`-агрегаты
- `median`-агрегаты
- blend `LightGBM + CatBoost`
- слишком точные временные агрегаты
- дополнительные устойчивые агрегаты по `month / is_weekend / dayofweek + month`

Краткие результаты собраны в [leaderboard_notes.md](./leaderboard_notes.md).

## Презентация

Ссылка на репозиторий должна быть размещена и в самой презентации:

- GitHub: [https://github.com/volodinmaksim/WB-hackaton-solo-track-](https://github.com/volodinmaksim/WB-hackaton-solo-track-)

## Проверка репозитория

Для проверки решения в репозитории есть:

- работающий код
- README
- отчёт
- конфиг финальной модели
- лучший сабмит
- презентация и её исходник
