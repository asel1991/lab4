"""tools.py - инструменты ReAct-агента.

calculator(expression)  - безопасный калькулятор на модуле ast (без eval!).
text_search(query, k)   - поиск по .txt файлам в папке docs/ (совпадение токенов).

Обе функции ВСЕГДА возвращают строку. Ошибки возвращаются как строка,
начинающаяся с "Error", чтобы агент мог увидеть их в Observation и
исправиться (исключения наружу не выбрасываются).
"""

import ast
import math
import operator
import re
from collections import Counter
from pathlib import Path

# ======================================================================
# calculator
# ======================================================================

# Белый список разрешённых операций: всё остальное запрещено.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPR_LEN = 200   # защита от слишком длинных выражений
_MAX_EXPONENT = 100   # защита от 9 ** 9 ** 9 и подобного


def _eval_node(node):
    """Рекурсивно вычисляет узел AST, разрешая только числа и + - * / // % **."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # bool - подкласс int, поэтому отсекаем его явно
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numbers are allowed")

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent is too large")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))

    # Имена, вызовы функций, атрибуты, списки, лямбды и т.д. - всё запрещено
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _normalize_expression(expr: str) -> str:
    """Приводит типичные 'человеческие' записи к валидному Python-выражению."""
    expr = str(expr).strip().strip("`").strip()
    if "=" in expr:                       # "2500*6 = 15000" -> "2500*6"
        expr = expr.split("=")[0].strip()
    expr = (
        expr.replace("\u00a0", " ")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )
    # "2 500 * 6" -> "2500 * 6" (пробел как разделитель тысяч)
    expr = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", expr)
    return expr


def _format_number(x) -> str:
    """Красиво форматирует результат: 15000.0 -> '15000', 0.1+0.2 -> '0.3'."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise ValueError("result is not a finite number")
        if x.is_integer():
            return str(int(x))
        return repr(round(x, 10))
    return str(x)


def calculator(expression: str) -> str:
    """Безопасно вычисляет арифметическое выражение и возвращает строку."""
    try:
        expr = _normalize_expression(expression)
        if not expr:
            return "Error: empty expression"
        if len(expr) > _MAX_EXPR_LEN:
            return f"Error: expression is longer than {_MAX_EXPR_LEN} characters"
        tree = ast.parse(expr, mode="eval")   # только выражения, не statements
        return _format_number(_eval_node(tree))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"Error: invalid expression ({exc})"


# ======================================================================
# text_search
# ======================================================================

DOCS_DIR = Path(__file__).resolve().parent / "docs"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    # русские
    "и", "в", "во", "на", "с", "со", "по", "за", "к", "о", "об", "от", "до", "для",
    "из", "у", "не", "что", "как", "это", "а", "но", "или", "при", "ли", "же", "бы",
    "сколько", "какой", "какая", "какие", "каков", "каково", "есть", "быть",
    # английские
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "what", "how",
}


def _stem(token: str) -> str:
    """Очень лёгкий 'стеммер': обрезаем длинные слова до 5 символов.

    Так 'отпуск', 'отпуска', 'отпуске' дают одну основу 'отпус'.
    Для учебного задания этого достаточно (можно заменить на pymorphy3).
    """
    return token[:5] if len(token) > 5 else token


def _tokenize(text: str) -> list:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        tokens.append(_stem(tok))
    return tokens


def _load_chunks(docs_dir: Path) -> list:
    """Читает все docs/*.txt и режет их на фрагменты по пустым строкам."""
    chunks = []
    for path in sorted(docs_dir.glob("*.txt")):
        # utf-8-sig: корректно читает файлы с BOM (частая проблема на Windows)
        text = path.read_text(encoding="utf-8-sig")
        for idx, para in enumerate(re.split(r"\n\s*\n", text), start=1):
            para = para.strip()
            if para:
                chunks.append(
                    {"source": path.name, "idx": idx, "text": para,
                     "tokens": set(_tokenize(para))}
                )
    return chunks


def text_search(query: str, top_k: int = 3) -> str:
    """Ищет top_k лучших фрагментов в docs/ по совпадению токенов.

    Оценка фрагмента = сумма IDF-весов общих токенов запроса и фрагмента
    (редкие слова весят больше). Возвращает фрагменты с указанием источника.
    """
    query = str(query).strip().strip("\"'`")
    if not query:
        return "Error: empty query"
    try:
        top_k = max(1, min(int(top_k), 10))
    except (TypeError, ValueError):
        top_k = 3

    if not DOCS_DIR.is_dir():
        return f"Error: docs directory not found: {DOCS_DIR}"
    chunks = _load_chunks(DOCS_DIR)
    if not chunks:
        return "Error: no .txt files found in docs/"

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return "No results: the query contains no meaningful words. Use specific keywords."

    n_chunks = len(chunks)
    doc_freq = Counter()
    for chunk in chunks:
        doc_freq.update(chunk["tokens"])   # tokens - множество: фрагмент считается 1 раз

    scored = []
    for chunk in chunks:
        common = q_tokens & chunk["tokens"]
        if common:
            score = sum(math.log(1 + n_chunks / doc_freq[t]) for t in common)
            scored.append((score, chunk))

    if not scored:
        return f"No results found for query '{query}'. Try different keywords."

    scored.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["idx"]))
    parts = []
    for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
        parts.append(
            f"[{rank}] Source: {chunk['source']} "
            f"(paragraph {chunk['idx']}, score {score:.2f})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    print(calculator("(1250 * 12 + 340) / 4"))
    print(text_search("суточные командировка Россия"))
