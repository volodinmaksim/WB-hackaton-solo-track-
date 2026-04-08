# Leaderboard Notes

## Текущий лучший результат

- Лучшая подтверждённая версия: `submission_route_time_agg_smooth_v1.csv`
- Public leaderboard score: `0.3608341250041473`
- Ключевая идея:
  - сглаженные агрегаты по маршруту и времени
  - `alpha = 20`

## Что уже проверили

### Count-агрегаты

- Public leaderboard score: `0.36776659947433227`
- Вывод:
  - хуже базовой smoothing-версии

### Median-агрегаты

- Локальный score: `0.40979012962272854`
- Вывод:
  - заметно хуже baseline

### Blend `LightGBM + CatBoost`

- Лучший локальный blend:
  - `LightGBM = 0.8`
  - `CatBoost = 0.2`
  - локальный score: `0.3834913913957258`
- Public leaderboard score:
  - `0.36264498745756096`
- Вывод:
  - blend не обогнал лучшую smoothing-версию

### Слишком точные временные агрегаты

- Что пробовали:
  - `(route_id, hour_slot, day)`
  - `(route_id, hour_slot, month)`
- Локальный score:
  - `0.46708208227641973`
- Вывод:
  - признаки оказались слишком шумными

### Более устойчивые дополнительные агрегаты

- Что пробовали:
  - `target_smooth_route_month`
  - `target_smooth_route_weekend`
  - `target_smooth_route_dow_month`
- Локальный score:
  - `0.4069738925952576`
- Вывод:
  - дополнительные сезонные агрегаты тоже оказались хуже текущей лучшей smoothing-версии

## Итог

- Лучшее решение финально зафиксировано как smoothing-версия с `alpha = 20`
- Новые эксперименты после неё не дали прироста на public leaderboard
