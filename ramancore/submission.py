"""Спектры от коллег для базы: описание, проверка, упаковка и отправка.

Как это работает:
  1. На сайте коллега загружает спектр и заполняет поля FIELDS.
  2. make_submission собирает «заявку»: спектр + описание (metadata.json)
     + исходный файл без изменений.
  3. Заявка уходит в закрытый репозиторий GitHub владельца базы (send_to_github).
  4. Владелец проверяет заявку и добавляет её в открытую базу скриптом
     scripts/import_submissions.py (строка для metadata — library_row).

Публикуются только поля, нужные базе; e-mail коллеги в открытую базу не попадает.
"""

import base64
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Описание полей формы — одно место и для сайта, и для проверки.
# kind: text, number, choice, check. group — раздел формы.
CLASSES = ["semiconductor", "carbon", "mineral", "polymer", "oxide", "other"]
SAMPLE_FORMS = ["монокристалл", "плёнка", "поликристалл", "порошок", "раствор или дисперсия",
                "наноструктуры", "другое"]
PROCESSING = ["исходный (без обработки)", "вычтен фон", "вычтен фон и сглажен", "другое"]

FIELDS = [
    # Образец
    {"key": "material", "label": "Материал", "group": "sample", "kind": "text", "required": True,
     "help": "Например: GaAs, графен, полистирол"},
    {"key": "formula", "label": "Химическая формула", "group": "sample", "kind": "text",
     "required": True, "help": "Например: GaAs, C, (C8H8)n"},
    {"key": "class", "label": "Класс материала", "group": "sample", "kind": "choice",
     "options": CLASSES, "required": True},
    {"key": "sample_form", "label": "Форма образца", "group": "sample", "kind": "choice",
     "options": SAMPLE_FORMS, "required": True},
    {"key": "phase_or_polytype", "label": "Фаза или политип", "group": "sample", "kind": "text",
     "help": "Например: 4H-SiC, вюрцит, анатаз"},
    {"key": "orientation", "label": "Ориентация поверхности", "group": "sample", "kind": "text",
     "help": "Например: (100), (111); для порошка — пусто"},
    {"key": "doping", "label": "Легирование, концентрация", "group": "sample", "kind": "text",
     "help": "Например: Te, n = 1.9·10¹⁸ см⁻³ (Холл)"},
    {"key": "substrate", "label": "Подложка", "group": "sample", "kind": "text"},
    {"key": "sample_id", "label": "Обозначение образца у вас", "group": "sample", "kind": "text"},
    # Съёмка
    {"key": "instrument", "label": "Прибор", "group": "measurement", "kind": "text",
     "required": True, "help": "Производитель и модель, например Confotec Duo"},
    {"key": "laser_nm", "label": "Длина волны лазера, нм", "group": "measurement",
     "kind": "number", "required": True, "min": 200.0, "max": 1100.0},
    {"key": "power_mW", "label": "Мощность на образце, мВт", "group": "measurement",
     "kind": "number", "min": 0.0, "max": 1000.0,
     "help": "Не процент, а мВт: процент от разных лазеров несравним"},
    {"key": "objective", "label": "Объектив", "group": "measurement", "kind": "text",
     "help": "Например: 100×, NA 0.9"},
    {"key": "grating", "label": "Решётка, штр/мм", "group": "measurement", "kind": "text"},
    {"key": "resolution_cm", "label": "Спектральное разрешение, см⁻¹", "group": "measurement",
     "kind": "number", "min": 0.0, "max": 100.0},
    {"key": "acquisition", "label": "Время накопления", "group": "measurement", "kind": "text",
     "help": "Например: 10 с × 5"},
    {"key": "polarization", "label": "Поляризация, геометрия", "group": "measurement",
     "kind": "text", "help": "Например: z(xx)z̄, без анализатора"},
    {"key": "temperature_K", "label": "Температура, К", "group": "measurement", "kind": "number",
     "min": 0.0, "max": 2000.0},
    {"key": "environment", "label": "Среда", "group": "measurement", "kind": "text",
     "help": "Воздух, вакуум, азот, раствор…"},
    {"key": "calibration", "label": "Калибровка", "group": "measurement", "kind": "text",
     "help": "Например: Si 520.7 см⁻¹ в тот же день"},
    {"key": "processing", "label": "Обработка спектра", "group": "measurement", "kind": "choice",
     "options": PROCESSING, "required": True},
    # Автор и источник
    {"key": "contributor", "label": "Ваше имя", "group": "source", "kind": "text", "required": True},
    {"key": "organization", "label": "Организация", "group": "source", "kind": "text"},
    {"key": "email", "label": "E-mail (не публикуется)", "group": "source", "kind": "text",
     "help": "Чтобы с вами связаться, если появятся вопросы по спектру"},
    {"key": "reference", "label": "Публикация (DOI)", "group": "source", "kind": "text",
     "help": "Если спектр опубликован"},
    {"key": "note", "label": "Комментарий", "group": "source", "kind": "text"},
    {"key": "license_ok", "label": "Согласен(на) на публикацию спектра в открытой базе по "
     "лицензии CC BY 4.0 (с указанием автора)", "group": "source", "kind": "check",
     "required": True},
    {"key": "rights_ok", "label": "Спектр снят мной или моей группой, и я вправе им делиться",
     "group": "source", "kind": "check", "required": True},
]
FIELD_KEYS = [f["key"] for f in FIELDS]
PRIVATE = {"email"}  # не попадают в открытую базу


def validate_submission(meta, x=None, y=None):
    """Проверяет описание (и спектр, если передан). Возвращает список ошибок."""
    errors = []
    for f in FIELDS:
        value = meta.get(f["key"])
        empty = value is None or (isinstance(value, str) and not value.strip())
        if f["kind"] == "check":
            if f.get("required") and value is not True:
                errors.append(f"Нужно согласие: «{f['label']}»")
            continue
        if empty:
            if f.get("required"):
                errors.append(f"Не заполнено: «{f['label']}»")
            continue
        if f["kind"] == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"«{f['label']}» должно быть числом")
                continue
            if not f.get("min", -np.inf) <= number <= f.get("max", np.inf):
                errors.append(f"«{f['label']}» вне диапазона {f['min']}–{f['max']}")
        if f["kind"] == "choice" and value not in f["options"]:
            errors.append(f"«{f['label']}»: недопустимое значение {value!r}")
    email = meta.get("email") or ""
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        errors.append("E-mail выглядит неверно")
    if x is not None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.size < 50:
            errors.append(f"В спектре всего {x.size} точек — нужно хотя бы 50")
        elif not np.all(np.isfinite(y)):
            errors.append("В спектре есть пустые или нечисловые значения")
        elif x.max() - x.min() < 200:
            errors.append("Диапазон спектра меньше 200 см⁻¹ — для базы мало")
    return errors


def _slug(text):
    """Короткая латиница/цифры для имени заявки: «GaAs (100)» → «gaas-100»."""
    text = re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower()
    return text[:30] or "sample"


def make_submission(x, y, meta, original_name=None, original_bytes=None, now=None):
    """Собирает заявку. Возвращает (sub_id, files), files — {имя: bytes}:
      spectrum.txt  — x и y через табуляцию (шапка # — комментарии);
      metadata.json — описание (все поля FIELDS + время, имя исходного файла);
      original<расширение> — исходный файл без изменений (если передан);
                       его настоящее имя (с пробелами, кириллицей) — в metadata.json.
    """
    now = now or datetime.now(timezone.utc)
    sub_id = f"{now:%Y%m%d-%H%M%S}-{_slug(meta.get('material'))}-{int(float(meta['laser_nm']))}"
    record = {k: meta.get(k) for k in FIELD_KEYS}
    record.update({"submission_id": sub_id, "submitted_utc": now.isoformat(timespec="seconds"),
                   "original_file": original_name, "n_points": int(len(x))})
    buf = io.StringIO()
    np.savetxt(buf, np.column_stack([x, y]), fmt="%.4f", delimiter="\t",
               header=f"{sub_id}: {meta.get('material')}, {meta.get('laser_nm')} нм")
    files = {"spectrum.txt": buf.getvalue().encode("utf-8"),
             "metadata.json": json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")}
    if original_bytes is not None and original_name:
        # Имя файла в заявке — только латиница: пробелы и кириллица ломают адрес в GitHub API
        suffix = re.sub(r"[^0-9a-zA-Z.]", "", Path(original_name).suffix)[:10]
        files[f"original{suffix}"] = original_bytes
    return sub_id, files


def send_to_github(sub_id, files, repo, token, folder="submissions", branch="main",
                   opener=urllib.request.urlopen):
    """Кладёт файлы заявки в репозиторий GitHub (папка folder/sub_id/)
    через REST API «create file contents». repo — «владелец/имя».
    token — fine-grained токен с правом Contents: write только на этот
    репозиторий. opener подменяется в тестах. Возвращает список путей."""
    paths = []
    for name, data in files.items():
        path = f"{folder}/{sub_id}/{name}"
        body = json.dumps({"message": f"Заявка {sub_id}: {name}", "branch": branch,
                           "content": base64.b64encode(data).decode("ascii")}).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(path)}",
            data=body, method="PUT",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json",
                     "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with opener(request, timeout=20) as response:
                if response.status not in (200, 201):
                    raise RuntimeError(f"GitHub ответил {response.status} на {path}")
        except urllib.error.HTTPError as error:
            hint = {401: "токен неверный или истёк",
                    403: "у токена нет права записи в этот репозиторий",
                    404: "репозиторий не найден или токен его не видит",
                    422: "заявка с таким именем уже есть"}.get(error.code, "")
            raise RuntimeError(f"GitHub ответил {error.code}" + (f": {hint}" if hint else "")) from error
        paths.append(path)
    return paths


def library_row(record, file):
    """Строка metadata.csv открытой базы из описания заявки (без приватных полей).
    Поля, которых нет в схеме базы, собираются в note."""
    main = {"id": "contrib_" + record["submission_id"].replace("-", "_"), "file": file,
            "material": record["material"], "formula": record["formula"],
            "class": record["class"], "laser_nm": float(record["laser_nm"]),
            "source": "contrib: " + ", ".join(v for v in (record.get("contributor"),
                                                         record.get("organization")) if v),
            "license": "CC BY 4.0"}
    for key in ("phase_or_polytype", "orientation", "power_mW", "instrument", "temperature_K"):
        main[key] = record.get(key)
    if record.get("reference"):
        ref = record["reference"]
        main["url"] = ref if ref.startswith("http") else f"https://doi.org/{ref}"
    used = set(main) | PRIVATE | {"license_ok", "rights_ok", "submission_id", "submitted_utc",
                                  "original_file", "n_points", "contributor", "organization",
                                  "reference"}
    labels = {f["key"]: f["label"] for f in FIELDS}
    extra = [f"{labels[k]}: {record[k]}" for k in FIELD_KEYS
             if k not in used and record.get(k) not in (None, "")]
    main["note"] = "; ".join(extra)
    return main
