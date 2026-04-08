const pptxgen = require("pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "WB Hackathon";
pptx.subject = "WB Hackathon Solo Track solution";
pptx.title = "WB Hackathon Solo Track — итоговое решение";
pptx.lang = "ru-RU";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "ru-RU",
};

function addTitle(slide, title, subtitle = "") {
  slide.addText(title, {
    x: 0.6, y: 0.4, w: 12.0, h: 0.5,
    fontFace: "Aptos Display", fontSize: 24, bold: true, color: "16324F"
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 0.95, w: 12.0, h: 0.35,
      fontFace: "Aptos", fontSize: 10.5, color: "52667A"
    });
  }
}

function addBullets(slide, items, opts = {}) {
  const runs = [];
  for (const item of items) {
    runs.push({ text: item, options: { bullet: { indent: 14 } } });
  }
  slide.addText(runs, {
    x: opts.x ?? 0.8,
    y: opts.y ?? 1.5,
    w: opts.w ?? 11.2,
    h: opts.h ?? 4.8,
    fontFace: "Aptos",
    fontSize: opts.fontSize ?? 18,
    color: "23384D",
    breakLine: true,
    margin: 0.05,
    paraSpaceAfterPt: 10,
    valign: "top",
  });
}

function finalize(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

{
  const slide = pptx.addSlide();
  slide.background = { color: "F3F7FA" };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.33, h: 0.28, fill: { color: "1E6F8C" }, line: { color: "1E6F8C" } });
  addTitle(slide, "WB Hackathon Solo Track", "Итоговое решение по прогнозу target_1h");
  slide.addText("Лучший public score: 0.3608341250041473", {
    x: 0.8, y: 2.0, w: 5.4, h: 0.5,
    fontFace: "Aptos Display", fontSize: 20, bold: true, color: "0E4B5A"
  });
  slide.addText("GitHub: https://github.com/volodinmaksim/WB-hackaton-solo-track-", {
    x: 0.8, y: 2.7, w: 10.8, h: 0.4,
    fontFace: "Aptos", fontSize: 14, color: "1E6F8C", hyperlink: { url: "https://github.com/volodinmaksim/WB-hackaton-solo-track-" }
  });
  addBullets(slide, [
    "Финальная модель использует только честные признаки: route_id, timestamp и агрегаты по train.",
    "Лучшее решение: LightGBM + сглаженные исторические агрегаты.",
    "Главный вывод: в этой задаче качество признаков важнее сложности модели."
  ], { y: 3.5, h: 2.2, fontSize: 18 });
  finalize(slide);
}

{
  const slide = pptx.addSlide();
  addTitle(slide, "Постановка задачи", "Что можно и нельзя использовать в финальной модели");
  addBullets(slide, [
    "Train: route_id, timestamp, status_1..6, target_1h.",
    "Test: id, route_id, timestamp.",
    "Нельзя использовать status_1..6 и прямые лаги target_1h, потому что их нет в test.",
    "Можно использовать календарные признаки и исторические агрегаты, рассчитанные только по train."
  ], { y: 1.5, h: 4.6 });
  finalize(slide);
}

{
  const slide = pptx.addSlide();
  addTitle(slide, "Ключевая идея", "Формула решения");
  slide.addText("f(route_id, timestamp, historical train patterns) → target_1h", {
    x: 0.9, y: 2.0, w: 11.2, h: 0.7,
    fontFace: "Aptos Display", fontSize: 24, bold: true, color: "16324F", align: "center"
  });
  addBullets(slide, [
    "Из timestamp извлекались hour, minute, hour_slot, dayofweek, day, month и weekend-признаки.",
    "Основной сигнал дали исторические паттерны маршрута по времени.",
    "Сырые средние работали хуже, чем сглаженные агрегаты."
  ], { y: 3.3, h: 2.6 });
  finalize(slide);
}

{
  const slide = pptx.addSlide();
  addTitle(slide, "Финальные признаки", "Что вошло в лучшую модель");
  addBullets(slide, [
    "Базовые признаки: route_id, hour, minute, hour_slot, dayofweek, day, month, is_weekend, is_month_start, is_month_end.",
    "Сглаженные агрегаты: route, route+hour_slot, route+dayofweek, route+hour_slot+dayofweek.",
    "Глобальные сглаженные агрегаты: hour_slot и dayofweek.",
    "Параметр сглаживания alpha = 20."
  ], { y: 1.5, h: 4.8 });
  finalize(slide);
}

{
  const slide = pptx.addSlide();
  addTitle(slide, "Результаты экспериментов", "Что сработало, а что нет");
  slide.addText("Сработало:", {
    x: 0.8, y: 1.5, w: 2.2, h: 0.3, fontFace: "Aptos Display", fontSize: 18, bold: true, color: "0E6B4F"
  });
  addBullets(slide, [
    "LightGBM + smoothing агрегатов",
    "alpha = 20",
    "Public leaderboard = 0.3608341250041473"
  ], { x: 0.8, y: 1.9, w: 5.4, h: 1.8, fontSize: 16 });

  slide.addText("Не сработало:", {
    x: 6.8, y: 1.5, w: 2.5, h: 0.3, fontFace: "Aptos Display", fontSize: 18, bold: true, color: "8A3B2E"
  });
  addBullets(slide, [
    "count-агрегаты",
    "median-агрегаты",
    "blend LightGBM + CatBoost",
    "слишком точные временные агрегаты"
  ], { x: 6.8, y: 1.9, w: 5.2, h: 2.6, fontSize: 16 });
  finalize(slide);
}

{
  const slide = pptx.addSlide();
  addTitle(slide, "Главные выводы", "Итог");
  addBullets(slide, [
    "Лучшая версия использует только честные признаки, доступные в test.",
    "Основной прирост дал feature engineering, а не усложнение модели.",
    "Исторические сглаженные агрегаты по маршруту и времени оказались самым сильным источником сигнала.",
    "Репозиторий содержит воспроизводимый код, README, отчёт и презентацию."
  ], { y: 1.6, h: 4.2 });
  finalize(slide);
}

pptx.writeFile({ fileName: "presentation/wb_hackathon_solution.pptx" });

