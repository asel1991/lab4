"""agent.py - чистый ReAct-агент (Yao et al., ICLR 2023) без фреймворков.

Цикл:  Thought -> Action + Action Input -> Observation -> ... -> Final Answer

Ключевые решения для Groq (чтобы не получать 400 tool_use_failed / 404):
  * НЕ передаём в API параметры tools / tool_choice - используем только текстовый протокол;
  * stop=["Observation:"] - модель не додумывает Observation сама (+ обрезка на клиенте);
  * модель выбирается автоматически из GET /models (список на Groq часто меняется),
    при 404 / model_decommissioned берётся следующий кандидат;
  * если сервер всё же вернул 400 tool_use_failed - берём текст из failed_generation
    и разбираем его нашим парсером, а не падаем;
  * для gpt-oss ставим reasoning_effort="low", у qwen3 вырезаем <think>...</think>.

Запуск:  python agent.py "Сколько суточных за 6 дней командировки по России?"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import (
    APIConnectionError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from tools import calculator, text_search

# Подхватываем GROQ_API_KEY из файла .env, если установлен python-dotenv
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Корректный вывод кириллицы в консоли Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ======================================================================
# Конфигурация
# ======================================================================

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LOG_PATH = Path(__file__).resolve().parent / "agent_log.jsonl"

# Порядок предпочтения: сначала обычные instruct-модели, gpt-oss - в конце,
# т.к. именно он провоцирует tool_use_failed. Реально используются только те,
# что есть в ответе GET /models. Принудительный выбор: переменная GROQ_MODEL.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
# Модели, которые не умеют в обычный чат (речь, модерация, встроенные агенты)
NON_CHAT_MARKERS = ("whisper", "guard", "tts", "orpheus", "playai", "compound",
                    "embed", "safeguard")

MAX_TOKENS = 1024            # у gpt-oss в лимит входят и токены рассуждений
MAX_RETRIES = 3              # повторы при 429 / сетевых ошибках
MAX_OBSERVATION_CHARS = 2000

LOG_FIELDS = ["ts", "run_id", "step", "event", "llm_output", "action",
              "action_input", "observation", "tool_ok", "latency_s", "usage"]

SYSTEM_PROMPT = """You are a corporate assistant that answers employee questions about internal company policies (HR, business travel, IT security, expenses).
You solve tasks with the ReAct method: you alternate between reasoning (Thought) and acting (Action).

Available actions:
1. text_search - full-text search over the company policy documents (the documents are in Russian).
   Action Input: a short query with 2-6 keywords in Russian, for example: суточные командировка Россия
   Returns the most relevant fragments with the source file name.
2. calculator - evaluates an arithmetic expression.
   Action Input: an expression such as (2500 * 6) + 100. Operators: + - * / // % ** and parentheses.

Use EXACTLY this plain-text format:

Thought: <your reasoning about what to do next>
Action: <one of: text_search, calculator>
Action Input: <the input for the action on a single line>

After writing Action Input, STOP. The system will run the action and reply with a message that starts with "Observation:". Then continue with a new Thought.

When you have all the information needed, finish with:

Thought: I now know the final answer
Final Answer: <the answer in the language of the question; mention the source file if you used documents>

Rules:
- Write exactly ONE Action per reply and never write "Observation:" yourself.
- Never do arithmetic in your head: always use the calculator.
- Use only facts from Observations. If the documents do not contain the answer, say so in the Final Answer (for example: "В предоставленных документах нет информации об этом") and do not invent anything.
- If an Observation starts with "Error", fix the input or choose another action.
- This is a plain-text protocol. Do NOT use native function/tool calling, JSON tool calls or code fences.
"""

# Реестр инструментов: имя действия -> функция
TOOLS = {
    "calculator": calculator,
    "text_search": text_search,
}

# ======================================================================
# Клиент и выбор модели
# ======================================================================

_client: OpenAI | None = None
_candidates: list | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it into the environment or into a .env file."
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def model_candidates() -> list:
    """Список моделей по порядку предпочтения (строится один раз за процесс)."""
    global _candidates
    if _candidates is not None:
        return _candidates

    try:
        available = [m.id for m in get_client().models.list().data]
    except Exception:
        available = []   # не получилось - пробуем модели по списку PREFERRED_MODELS

    chat_models = [m for m in available
                   if not any(mark in m.lower() for mark in NON_CHAT_MARKERS)]

    ordered = []
    forced = os.getenv("GROQ_MODEL")
    if forced:
        ordered.append(forced)
    if chat_models:
        ordered += [m for m in PREFERRED_MODELS if m in chat_models]
        ordered += sorted(chat_models)
    else:
        ordered += PREFERRED_MODELS

    _candidates = list(dict.fromkeys(ordered))   # убираем дубли, сохраняя порядок
    return _candidates


# ======================================================================
# Вызов LLM
# ======================================================================

def _error_info(exc) -> dict:
    """Достаёт словарь ошибки ({'message','type','code','failed_generation'})."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        return inner if isinstance(inner, dict) else body
    try:
        data = exc.response.json()
        inner = data.get("error")
        return inner if isinstance(inner, dict) else data
    except Exception:
        return {}


def _retry_delay(exc, attempt: int) -> float:
    try:
        return min(float(exc.response.headers.get("retry-after")), 30.0)
    except Exception:
        return 5.0 * (attempt + 1)


def strip_think(text: str) -> str:
    """Удаляет блоки <think>...</think> (reasoning-модели вроде qwen3)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def cut_at_observation(text: str) -> str:
    """Обрезает текст по 'Observation:' (страховка к параметру stop)."""
    match = re.search(r"(?:^|\n)\s*Observation\s*:", text)
    return text[: match.start()] if match else text


def call_llm(messages: list) -> dict:
    """Вызывает чат-модель. Возвращает {text, usage, model, latency_s, recovered}.

    Перебирает модели-кандидаты; обрабатывает 404, 429, сетевые ошибки и
    400 tool_use_failed (берёт failed_generation как ответ модели).
    """
    client = get_client()
    last_exc = None

    for model in list(model_candidates()):
        kwargs = dict(model=model, messages=messages, temperature=0,
                      max_tokens=MAX_TOKENS, stop=["Observation:"])
        if "gpt-oss" in model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}

        for attempt in range(MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)

            except NotFoundError as exc:                      # 404 model_not_found
                last_exc = exc
                break                                         # -> следующая модель

            except RateLimitError as exc:                     # 429
                last_exc = exc
                time.sleep(_retry_delay(exc, attempt))
                continue

            except APIConnectionError as exc:                 # сеть
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
                continue

            except BadRequestError as exc:                    # 400
                info = _error_info(exc)
                code = info.get("code")
                if code == "tool_use_failed" and info.get("failed_generation"):
                    # Сервер "перехватил" наш текст Action как вызов функции.
                    # Достаём исходный текст модели и разбираем его сами.
                    return {
                        "text": strip_think(str(info["failed_generation"])),
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0},
                        "model": model,
                        "latency_s": round(time.perf_counter() - t0, 3),
                        "recovered": True,
                    }
                if code in ("model_decommissioned", "model_not_found"):
                    last_exc = exc
                    break                                     # -> следующая модель
                if "stop" in kwargs and "stop" in str(exc).lower():
                    kwargs.pop("stop")                        # модель не принимает stop
                    continue
                raise

            else:
                latency = round(time.perf_counter() - t0, 3)
                u = getattr(resp, "usage", None)
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }
                text = strip_think(resp.choices[0].message.content or "")
                # запоминаем рабочую модель: в следующий раз пробуем её первой
                cands = model_candidates()
                if cands and cands[0] != model:
                    cands.remove(model)
                    cands.insert(0, model)
                return {"text": text, "usage": usage, "model": model,
                        "latency_s": latency, "recovered": False}

    raise RuntimeError(f"No working model found. Last error: {last_exc}")


# ======================================================================
# Парсер ответа модели (регулярные выражения)
# ======================================================================

ACTION_RE = re.compile(r"Action\s*:\s*([^\n\r]+)", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(
    r"Action\s+Input\s*:\s*(.*?)(?=\n\s*(?:Thought|Observation|Action|Final\s+Answer)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_RE = re.compile(r"Final\s+Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)

# Запасные форматы "нативного" вызова функции (на случай failed_generation)
_HARMONY_RE = re.compile(r"to=(?:functions\.)?([A-Za-z_]\w*).*?(\{.*\})", re.DOTALL)
_FUNC_TAG_RE = re.compile(r"<function=([A-Za-z_]\w*)[^>]*>\s*(\{.*?\})", re.DOTALL)


def _clean_action_name(name: str) -> str:
    name = name.strip().strip("`'\"*").strip()
    name = re.sub(r"\(.*\)\s*$", "", name)        # calculator() -> calculator
    return name.strip().strip("`'\"*").strip()


def _clean_action_input(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```\w*\s*|\s*```$", "", value).strip()    # убираем ``` ```
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
        value = value[1:-1].strip()
    return value


def parse_llm_output(text: str) -> dict:
    """Разбирает ответ модели.

    Возвращает один из вариантов:
      {"kind": "final",  "answer": str}
      {"kind": "action", "action": str, "action_input": str}
      {"kind": "invalid", "reason": str}
    Если в тексте есть и Action, и Final Answer, побеждает тот, что встретился раньше.
    """
    text = cut_at_observation(text)
    m_act = ACTION_RE.search(text)
    m_fin = FINAL_RE.search(text)

    if m_act and (not m_fin or m_act.start() < m_fin.start()):
        m_in = ACTION_INPUT_RE.search(text, m_act.end())
        if m_in:
            return {"kind": "action",
                    "action": _clean_action_name(m_act.group(1)),
                    "action_input": _clean_action_input(m_in.group(1))}
        return {"kind": "invalid", "reason": "'Action' without 'Action Input'"}

    if m_fin and m_fin.group(1).strip():
        return {"kind": "final", "answer": m_fin.group(1).strip()}

    # запасной вариант: модель выдала "нативный" вызов функции
    for regex in (_HARMONY_RE, _FUNC_TAG_RE):
        m_native = regex.search(text)
        if m_native:
            return {"kind": "action", "action": m_native.group(1),
                    "action_input": m_native.group(2).strip()}

    return {"kind": "invalid", "reason": "neither Action nor Final Answer found"}


# ======================================================================
# Выполнение инструментов
# ======================================================================

def _prepare_kwargs(tool_name: str, raw: str) -> dict:
    """Превращает Action Input в аргументы функции (понимает и JSON, и обычный текст)."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if tool_name == "calculator":
                value = next((data[k] for k in ("expression", "expr", "input", "query")
                              if k in data), None)
                if value is None and data:
                    value = next(iter(data.values()))
                return {"expression": "" if value is None else str(value)}
            value = next((data[k] for k in ("query", "q", "text", "input") if k in data), None)
            if value is None and data:
                value = next(iter(data.values()))
            kwargs = {"query": "" if value is None else str(value)}
            if "top_k" in data:
                kwargs["top_k"] = data["top_k"]
            return kwargs
    return {"expression": raw} if tool_name == "calculator" else {"query": raw}


def execute_action(action: str, action_input: str) -> tuple:
    """Запускает инструмент. Возвращает (observation: str, tool_ok: bool)."""
    name = action.strip().lower()
    tool = TOOLS.get(name)
    if tool is None:
        return (f"Error: unknown action '{action}'. "
                f"Available actions: {', '.join(TOOLS)}.", False)
    try:
        observation = str(tool(**_prepare_kwargs(name, action_input)))
    except Exception as exc:                                  # инструмент не должен ронять агента
        observation = f"Error: {type(exc).__name__}: {exc}"
    return observation, not observation.startswith("Error")


# ======================================================================
# Логирование (JSON Lines)
# ======================================================================

def log_event(run_id: str, step: int, event: str, **fields) -> None:
    """Пишет одну строку в agent_log.jsonl. Все обязательные поля присутствуют всегда."""
    record = {key: None for key in LOG_FIELDS}
    record.update(
        ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        run_id=run_id, step=step, event=event,
    )
    record.update(fields)    # дополнительные поля (question, model, final_answer, ...)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ======================================================================
# Главная функция
# ======================================================================

def run_agent(question: str, max_steps: int = 8, run_id: str | None = None,
              verbose: bool = True) -> str:
    """Запускает ReAct-цикл и возвращает финальный ответ (строку).

    События в agent_log.jsonl:
      run_start      - начало (поле question)
      llm_response   - ответ модели: llm_output, latency_s, usage
      tool_call      - action, action_input, observation, tool_ok, latency_s (время инструмента)
      parse_error    - модель нарушила формат (observation = подсказка модели)
      final_answer   - итог (final_answer, run_usage_total, total_latency_s)
      llm_error / max_steps_reached - аварийные завершения
    """
    run_id = run_id or uuid.uuid4().hex[:8]
    t_run = time.perf_counter()
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    log_event(run_id, 0, "run_start", question=question, max_steps=max_steps)
    if verbose:
        print(f"\n=== run {run_id} ===\nQuestion: {question}")

    for step in range(1, max_steps + 1):
        # ---- 1. Thought + Action от модели -------------------------------
        try:
            llm = call_llm(messages)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            log_event(run_id, step, "llm_error", observation=message,
                      total_latency_s=round(time.perf_counter() - t_run, 3))
            if verbose:
                print(f"[step {step}] LLM error: {message}")
            return f"Ошибка при обращении к LLM: {message}"

        text = cut_at_observation(llm["text"]).strip()
        for key in totals:
            totals[key] += llm["usage"].get(key, 0)
        log_event(run_id, step, "llm_response", llm_output=text,
                  latency_s=llm["latency_s"], usage=llm["usage"],
                  model=llm["model"], recovered_from_tool_use_failed=llm["recovered"])
        if verbose:
            print(f"[step {step}] ({llm['model']})\n{text}")

        parsed = parse_llm_output(text)

        # ---- 2a. Final Answer --------------------------------------------
        if parsed["kind"] == "final":
            log_event(run_id, step, "final_answer", llm_output=text,
                      final_answer=parsed["answer"], run_usage_total=totals,
                      total_latency_s=round(time.perf_counter() - t_run, 3))
            if verbose:
                print(f"=> Final Answer: {parsed['answer']}")
            return parsed["answer"]

        # ---- 2b. Action -> Observation -----------------------------------
        if parsed["kind"] == "action":
            t_tool = time.perf_counter()
            observation, tool_ok = execute_action(parsed["action"], parsed["action_input"])
            observation = observation[:MAX_OBSERVATION_CHARS]
            log_event(run_id, step, "tool_call", llm_output=text,
                      action=parsed["action"], action_input=parsed["action_input"],
                      observation=observation, tool_ok=tool_ok,
                      latency_s=round(time.perf_counter() - t_tool, 4))
            if verbose:
                shown = observation if len(observation) < 300 else observation[:300] + "..."
                print(f"Observation (ok={tool_ok}): {shown}")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # ---- 2c. Формат нарушен -> просим модель исправиться ---------------
        hint = ("Observation: Invalid format. Reply with 'Thought:' and then EITHER "
                "'Action: <text_search|calculator>' followed by 'Action Input: <input>' "
                "OR 'Final Answer: <answer>'.")
        log_event(run_id, step, "parse_error", llm_output=text, observation=hint,
                  tool_ok=False)
        if verbose:
            print(f"Parse error: {parsed['reason']}")
        messages.append({"role": "assistant", "content": text[:1000] or "(empty)"})
        messages.append({"role": "user", "content": hint})

    # ---- Лимит шагов исчерпан --------------------------------------------
    log_event(run_id, max_steps, "max_steps_reached", run_usage_total=totals,
              total_latency_s=round(time.perf_counter() - t_run, 3))
    if verbose:
        print("=> max_steps reached")
    return f"Не удалось получить ответ за {max_steps} шагов."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Сколько календарных дней ежегодного отпуска положено сотруднику?"
    print("\nИтог:", run_agent(q))
