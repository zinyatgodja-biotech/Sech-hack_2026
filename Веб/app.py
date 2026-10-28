"""MicroFlow: небольшой сервер для учебного прототипа.
Пока все движения виртуальные; Arduino НЕ подключена.
"""
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
PROGRAM_FILE = Path(__file__).with_name("program.json")

# Это состояние демонстрационного стенда. УСЛОВНЫЕ ЗНАЧЕНИЯ!!!!.
state = {
    "position": 0.0,
    "mode": "Ожидание",
    "stage": "—",
    "cycle": 0,
    "total_cycles": 0,
    "connected": False,
    "message": "Готов к симуляции",
}
lock = threading.Lock()
stop_event = threading.Event()
worker = None


def get_state():
    """Вернуть текущее состояние для интерфейса."""
    with lock:
        return dict(state)


def update_state(**changes):
    with lock:
        state.update(changes)


def check_number(value, name, minimum, maximum):
    """Проверить числовое поле, не пропуская NaN и бесконечность."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Некорректное поле: {name}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}: значение должно быть от {minimum} до {maximum}")
    return value


def read_program(data):
    """Проверить параметры программы (условные пределы симулятора)."""
    cycles = check_number(data.get("cycles"), "Циклы", 1, 50)
    if not cycles.is_integer():
        raise ValueError("Количество циклов должно быть целым числом")
    return {
        "travel": check_number(data.get("travel"), "Ход", 0.1, 10),
        "speed": check_number(data.get("speed"), "Скорость", 0.1, 5),
        "hold": check_number(data.get("hold"), "Выдержка", 0, 30),
        "cycles": int(cycles),
    }


def wait_with_stop(seconds):
    """Ожидать указанное время, но дать возможность быстро остановить цикл."""
    return not stop_event.wait(seconds)


def move_smoothly(target, speed):
    """Симуляция перемещения. Реальным мотором эта функция НЕ управляет."""
    while not stop_event.is_set():
        current = get_state()["position"]
        delta = target - current
        if abs(delta) <= 0.01:
            update_state(position=round(target, 2))
            return True
        increment = min(abs(delta), speed * 0.05)
        new_position = current + (increment if delta > 0 else -increment)
        update_state(position=round(new_position, 2))
        if not wait_with_stop(0.05):
            break
    return False


def execute_program(program):
    """Выполнить в фоне виртуальные циклы нажать -> подождать -> отпустить."""
    try:
        for cycle in range(1, program["cycles"] + 1):
            if stop_event.is_set():
                break
            update_state(cycle=cycle, stage="Нажатие", message="Плунжер движется вниз")
            if not move_smoothly(program["travel"], program["speed"]):
                break
            update_state(stage="Выдержка", message="Плунжер удерживается")
            if not wait_with_stop(program["hold"]):
                break
            update_state(stage="Отжатие", message="Плунжер возвращается")
            if not move_smoothly(0, program["speed"]):
                break
        if stop_event.is_set():
            update_state(mode="Остановлено", stage="—", message="Симуляция остановлена")
        else:
            update_state(mode="Готово", stage="—", message="Все циклы завершены")
    except Exception:
        update_state(mode="Ошибка", stage="—", message="Ошибка при выполнении")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(get_state())


@app.post("/api/move")
def move():
    """Ручное перемещение виртуального плунжера."""
    if get_state()["mode"] in ("Работает", "Останавливается"):
        return jsonify(error="Сначала остановите программу"), 409
    try:
        target = check_number(request.json.get("position"), "Положение", 0, 10)
    except (ValueError, AttributeError) as error:
        return jsonify(error=str(error)), 400
    update_state(position=round(target, 2), mode="Ручной", stage="—", message="Положение изменено")
    return jsonify(get_state())


@app.post("/api/start")
def start():
    global worker
    if get_state()["mode"] in ("Работает", "Останавливается") or (worker and worker.is_alive()):
        return jsonify(error="Подождите завершения предыдущей симуляции"), 409
    try:
        program = read_program(request.json or {})
    except (ValueError, AttributeError) as error:
        return jsonify(error=str(error)), 400
    stop_event.clear()
    update_state(mode="Работает", cycle=0, total_cycles=program["cycles"], message="Симуляция запущена")
    worker = threading.Thread(target=execute_program, args=(program,), daemon=True)
    worker.start()
    return jsonify(get_state())


@app.post("/api/stop")
def stop():
    stop_event.set()
    if get_state()["mode"] == "Работает":
        update_state(mode="Останавливается", message="Остановка симуляции...")
    return jsonify(get_state())


@app.get("/api/program")
def load_program():
    if not PROGRAM_FILE.exists():
        return jsonify(error="Сохранённой программы пока нет"), 404
    return jsonify(json.loads(PROGRAM_FILE.read_text(encoding="utf-8")))


@app.post("/api/program")
def save_program():
    try:
        program = read_program(request.json or {})
    except (ValueError, AttributeError) as error:
        return jsonify(error=str(error)), 400
    PROGRAM_FILE.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify(message="Программа сохранена")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
