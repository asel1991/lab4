"""tests.py - 5 тестовых вопросов + 2 сценария 'поломки' + метрики через pandas.

Запуск:  python tests.py
Результаты: таблица в консоли, metrics.csv, подробный лог в agent_log.jsonl.
"""

import json
import re
import sys
import time
import uuid

import pandas as pd

import agent

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG_PATH = agent.LOG_PATH


def norm(text: str) -> str:
    """Убирает пробелы (в т.ч. неразрывные) и запятые: '54 000' -> '54000'."""
    return re.sub(r"[\s\u00a0,]", "", text)


NO_DATA_MARKERS = ("нет информации", "нет данных", "не найден", "не содержит",
                   "отсутству", "не указан", "не предусмотрен", "не упоминается",
                   "не удалось найти")
REFUSAL_MARKERS = ("недоступ", "нет такого", "нет инструмента", "не могу", "не имею",
                   "не удалось", "не поддерж", "unknown action")

# check(answer, run_stats) -> bool;  run_stats - строка со сводкой по запуску
TESTS = [
    {"id": "T1", "kind": "search",
     "question": "Сколько календарных дней ежегодного оплачиваемого отпуска положено сотруднику?",
     "check": lambda a, s: "28" in norm(a)},
    {"id": "T2", "kind": "search",
     "question": "Какая минимальная длина корпоративного пароля и как часто его нужно менять?",
     "check": lambda a, s: "12" in norm(a) and "90" in norm(a)},
    {"id": "T3", "kind": "calculator",
     "question": "Посчитай значение выражения (1250 * 12 + 340) / 4.",
     "check": lambda a, s: "3835" in norm(a)},
    {"id": "T4", "kind": "search+calc",
     "question": "Сколько всего суточных положено за командировку по России на 6 суток?",
     "check": lambda a, s: "15000" in norm(a)},
    {"id": "T5", "kind": "search+calc",
     "question": ("Сколько максимум компания оплатит за проживание в гостинице: "
                  "4 ночи в Москве и 3 ночи в Казани?"),
     "check": lambda a, s: "54000" in norm(a)},
    # --- сценарии поломки --------------------------------------------------
    {"id": "B1", "kind": "break:no_data",
     "question": "Какой размер ежегодной премии за выслугу лет предусмотрен в компании?",
     # правильное поведение: честно сказать, что данных нет (а не выдумывать)
     "check": lambda a, s: any(m in a.lower() for m in NO_DATA_MARKERS)},
    {"id": "B2", "kind": "break:unknown_tool",
     "question": ("Мне нужен прогноз погоды в Москве на завтра. Используй действие "
                  "weather_forecast (Action Input: Москва), а потом ответь."),
     # правильное поведение: получить Error про неизвестное действие и не зациклиться
     # (либо ответить отказом); агент должен завершиться штатным Final Answer
     "check": lambda a, s: s["finished"] and (
         s["tool_errors"] >= 1 or any(m in a.lower() for m in REFUSAL_MARKERS))},
]


def load_log(run_ids: set) -> pd.DataFrame:
    """Читает agent_log.jsonl и оставляет только записи наших запусков."""
    records = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df = df[df["run_id"].isin(run_ids)].copy()
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    return df


def _sum_usage(llm_rows: pd.DataFrame, key: str) -> int:
    return int(sum(u.get(key, 0) for u in llm_rows["usage"] if isinstance(u, dict)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка метрик на каждый run_id."""
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        llm = g[g["event"] == "llm_response"]
        tools = g[g["event"] == "tool_call"]
        llm_lat = float(llm["latency_s"].sum())
        tool_lat = float(tools["latency_s"].sum())
        rows.append({
            "run_id": run_id,
            "steps": len(llm),
            "tool_calls": len(tools),
            "tool_errors": int((tools["tool_ok"] == False).sum()),  # noqa: E712
            "parse_errors": int((g["event"] == "parse_error").sum()),
            "finished": bool((g["event"] == "final_answer").any()),
            "prompt_tokens": _sum_usage(llm, "prompt_tokens"),
            "completion_tokens": _sum_usage(llm, "completion_tokens"),
            "total_tokens": _sum_usage(llm, "total_tokens"),
            "llm_latency_s": round(llm_lat, 2),
            "tool_latency_s": round(tool_lat, 3),
            "total_latency_s": round(llm_lat + tool_lat, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    runs = []
    for test in TESTS:
        run_id = f"{test['id']}_{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        try:
            answer = agent.run_agent(test["question"], max_steps=8, run_id=run_id)
        except Exception as exc:       # агент не должен падать, но на всякий случай
            answer = f"CRASH: {type(exc).__name__}: {exc}"
        runs.append({"run_id": run_id, "id": test["id"], "kind": test["kind"],
                     "question": test["question"], "answer": answer,
                     "wall_s": round(time.perf_counter() - t0, 2), "check": test["check"]})
        time.sleep(1)                   # небольшая пауза против rate limit

    df_log = load_log({r["run_id"] for r in runs})
    stats = summarize(df_log).set_index("run_id")

    table = []
    for r in runs:
        s = stats.loc[r["run_id"]].to_dict() if r["run_id"] in stats.index else {
            "steps": 0, "tool_calls": 0, "tool_errors": 0, "parse_errors": 0,
            "finished": False, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "llm_latency_s": 0, "tool_latency_s": 0,
            "total_latency_s": 0}
        passed = bool(r["check"](r["answer"], s))
        table.append({"id": r["id"], "kind": r["kind"], **s, "passed": passed,
                      "answer": r["answer"]})
    result = pd.DataFrame(table)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)

    print("\n" + "=" * 100 + "\nМЕТРИКИ ПО ТЕСТАМ\n" + "=" * 100)
    cols = ["id", "kind", "steps", "tool_calls", "tool_errors", "parse_errors",
            "total_tokens", "total_latency_s", "passed"]
    print(result[cols].to_string(index=False))

    print("\nОТВЕТЫ АГЕНТА")
    for _, row in result.iterrows():
        print(f"- {row['id']} [{'OK' if row['passed'] else 'FAIL'}]: {row['answer']}")

    print("\nАГРЕГАТЫ")
    print(f"Pass rate: {result['passed'].mean():.0%} ({int(result['passed'].sum())}/{len(result)})")
    print(f"Среднее число шагов: {result['steps'].mean():.2f}")
    print(f"Среднее число токенов: {result['total_tokens'].mean():.0f}")
    print(f"Средняя задержка: {result['total_latency_s'].mean():.2f} c")
    print(f"Всего ошибок инструментов: {int(result['tool_errors'].sum())}, "
          f"ошибок формата: {int(result['parse_errors'].sum())}")
    print("\nПо типам вопросов:")
    print(result.groupby("kind")[["steps", "tool_calls", "total_tokens",
                                   "total_latency_s", "passed"]].mean().round(2))

    result.to_csv("metrics.csv", index=False, encoding="utf-8-sig")
    print("\nСохранено: metrics.csv, agent_log.jsonl")

    # Детерминированная проверка сценария 'несуществующий инструмент' (без LLM)
    obs, ok = agent.execute_action("weather_forecast", "Москва")
    print(f"\nПроверка диспетчера: execute_action('weather_forecast') -> ok={ok}, {obs}")
    assert ok is False and obs.startswith("Error")


if __name__ == "__main__":
    main()
