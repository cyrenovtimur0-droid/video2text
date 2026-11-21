
import os
import json
import time
import uuid
import subprocess
import shutil
from datetime import timedelta

import yt_dlp
from urllib.parse import urlparse

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import whisper

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect,
    url_for
)

from werkzeug.utils import secure_filename


# -------------------------------
# НАСТРОЙКИ FFMPEG
# -------------------------------

FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

os.environ["PATH"] = (
    os.path.dirname(FFMPEG_PATH)
    + os.pathsep
    + os.environ.get("PATH", "")
)


# -------------------------------
# НАСТРОЙКИ FLASK
# -------------------------------

app = Flask(__name__)

# Постоянный ключ нужен, чтобы сессия
# не сбрасывалась при перезапуске сервера.
# Это ключ для локальной разработки.
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "video2text-local-development-key"
)

app.permanent_session_lifetime = timedelta(days=7)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# -------------------------------
# ОЧИСТКА СТАРЫХ ФАЙЛОВ
# -------------------------------

def cleanup_old_files():
    now = time.time()

    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)

            # Удаляем файлы старше 7 дней
            if file_age > 604800:
                os.remove(filepath)


# -------------------------------
# ЗАГРУЗКА WHISPER
# -------------------------------

print("Загрузка модели Whisper...")

model = whisper.load_model("small")

print("Модель Whisper загружена!")

cleanup_old_files()


# -------------------------------
# ГЛАВНАЯ СТРАНИЦА
# -------------------------------

@app.route("/")
def home():
    result_id = session.get("result_id")
    saved_result = None

    if result_id:
        result_path = os.path.join(
            UPLOAD_FOLDER,
            f"{result_id}.json"
        )

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as file:
                saved_result = json.load(file)

    return render_template(
        "index.html",
        segments=saved_result["segments"] if saved_result else None,
        full_text=saved_result["full_text"] if saved_result else None,
        uploaded_filename=session.get("uploaded_filename")
    )


# -------------------------------
# СТРАНИЦА ТАРИФОВ
# -------------------------------

@app.route("/tariffs")
def tariffs():
    return render_template("tariffs.html")


# -------------------------------
# СОХРАНЕНИЕ ИЗМЕНЁННОГО ТЕКСТА
# -------------------------------

@app.route("/save_text", methods=["POST"])
def save_text():
    result_id = session.get("result_id")

    if not result_id:
        return "Результат не найден"

    result_path = os.path.join(
        UPLOAD_FOLDER,
        f"{result_id}.json"
    )

    if not os.path.exists(result_path):
        return "Файл результата не найден"

    new_text = request.form.get("recognized_text", "")

    with open(result_path, "r", encoding="utf-8") as file:
        saved_result = json.load(file)

    saved_result["full_text"] = new_text

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            saved_result,
            file,
            ensure_ascii=False
        )

    download_path = os.path.join(
        UPLOAD_FOLDER,
        f"{result_id}.txt"
    )

    with open(download_path, "w", encoding="utf-8") as file:
        file.write(new_text)

    return redirect(url_for("home"))


# -------------------------------
# РАСПОЗНАВАНИЕ ВИДЕО И АУДИО
# -------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():

    if "file" not in request.files:
        return "Файл не найден"

    file = request.files["file"]

    if file.filename == "":
        return "Выберите видео или аудио"

    original_filename = secure_filename(file.filename)

    if not original_filename:
        return "Некорректное имя файла"

    # Уникальное имя, чтобы загрузки не перезаписывали друг друга
    upload_id = uuid.uuid4().hex

    filepath = os.path.join(
        UPLOAD_FOLDER,
        f"{upload_id}_{original_filename}"
    )

    file.save(filepath)

    processed_filepath = os.path.join(
        UPLOAD_FOLDER,
        f"{upload_id}_clean.wav"
    )

    try:

        # -------------------------------
        # ОБРАБОТКА ЗВУКА ЧЕРЕЗ FFMPEG
        # -------------------------------

        subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", filepath,
                "-af",
"highpass=f=100,lowpass=f=7500,afftdn=nf=-20",
                "-ac", "1",
                "-ar", "16000",
                processed_filepath
            ],
            check=True
        )

        # -------------------------------
        # РАСПОЗНАВАНИЕ WHISPER
        # -------------------------------

        result = model.transcribe(
            processed_filepath,
            language="ru",
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            initial_prompt=(
                'Русская разговорная речь. '
                'Правильные слова: Хельсинки, девчонки. '
                'Распознавай только произнесённые слова.'
            )
        )

        segments = result["segments"]

        # -------------------------------
        # ИСПРАВЛЕНИЕ ЧАСТЫХ ОШИБОК
        # -------------------------------

        corrections = {
            "Девчонги": "девчонки",
            "девчонги": "девчонки",
            "Хельсинги": "Хельсинки",
            "хельсинги": "Хельсинки",
            "подвижно": "подвинусь",
            "Подвижно": "Подвинусь",
            "карангарск": "Ангарск",
            "Карангарск": "Ангарск",
        }

        for segment in segments:
            for wrong, correct in corrections.items():
                segment["text"] = segment["text"].replace(
                    wrong,
                    correct
                )

        # -------------------------------
        # ПОДГОТОВКА ТЕКСТА
        # -------------------------------

        text = "\n".join(
            segment["text"].strip()
            for segment in segments
        )

        result_id = uuid.uuid4().hex

        result_path = os.path.join(
            UPLOAD_FOLDER,
            f"{result_id}.json"
        )

        saved_segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in segments
        ]

        # -------------------------------
        # СОХРАНЕНИЕ РЕЗУЛЬТАТА
        # -------------------------------

        with open(result_path, "w", encoding="utf-8") as json_file:
            json.dump(
                {
                    "segments": saved_segments,
                    "full_text": text
                },
                json_file,
                ensure_ascii=False
            )

        session.permanent = True
        session["result_id"] = result_id
        session["uploaded_filename"] = original_filename

        # -------------------------------
        # СОЗДАНИЕ TXT
        # -------------------------------

        download_name = f"{result_id}.txt"

        download_path = os.path.join(
            UPLOAD_FOLDER,
            download_name
        )

        with open(download_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

        session["download_file"] = download_name

        return render_template(
            "index.html",
            segments=saved_segments,
            full_text=text,
            uploaded_filename=original_filename
        )

    except Exception as e:

        return f"Ошибка распознавания: {e}"

    finally:

        # Удаляем обработанный WAV,
        # но сохраняем исходное видео или аудио.
        if os.path.exists(processed_filepath):
            os.remove(processed_filepath)


# -------------------------------
# РАСПОЗНАВАНИЕ ПО ССЫЛКЕ
# -------------------------------

@app.route("/transcribe_url", methods=["POST"])
def transcribe_url():

    video_url = request.form.get("video_url", "").strip()

    if not video_url:
        return "Вставьте ссылку на видео"

    parsed_url = urlparse(video_url)

    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        return "Некорректная ссылка"

    download_id = uuid.uuid4().hex

    output_template = os.path.join(
        UPLOAD_FOLDER,
        f"{download_id}.%(ext)s"
    )

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_file = ydl.prepare_filename(info)

        if not os.path.exists(downloaded_file):
            return "Не удалось получить аудио с этой платформы"

        processed_filepath = os.path.join(
            UPLOAD_FOLDER,
            f"{download_id}_clean.wav"
        )

        subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", downloaded_file,
                "-af", "highpass=f=100,lowpass=f=7500,afftdn=nf=-20",
                "-ac", "1",
                "-ar", "16000",
                processed_filepath
            ],
            check=True,
            capture_output=True,
            text=True
        )

        result = model.transcribe(
            processed_filepath,
            language="ru",
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )

        segments = result["segments"]

        text = "\n".join(
            segment["text"].strip()
            for segment in segments
        )

        result_id = uuid.uuid4().hex

        saved_segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in segments
        ]

        result_path = os.path.join(
            UPLOAD_FOLDER,
            f"{result_id}.json"
        )

        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "segments": saved_segments,
                    "full_text": text
                },
                file,
                ensure_ascii=False
            )

        download_name = f"{result_id}.txt"

        with open(
            os.path.join(UPLOAD_FOLDER, download_name),
            "w",
            encoding="utf-8"
        ) as file:
            file.write(text)

        session.permanent = True
        session["result_id"] = result_id
        session["download_file"] = download_name
        session["uploaded_filename"] = info.get("title", "Видео по ссылке")

        return redirect(url_for("home"))

    except Exception:
        return (
            "Не удалось получить видео. Возможно, платформа "
            "не поддерживается, видео недоступно или требует авторизации."
        )

    finally:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.startswith(download_id):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)


# -------------------------------
# СКАЧИВАНИЕ TXT
# -------------------------------

@app.route("/download")
def download():
    download_file = session.get("download_file")

    if not download_file:
        return "Файл не найден"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        download_file
    )

    if not os.path.exists(filepath):
        return "Файл не найден"

    return send_file(
        filepath,
        as_attachment=True,
        download_name="transcription.txt",
        mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)