"""Evaluation metrics for OCR output quality."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance — lower is better (0 = perfect)."""
    import editdistance
    p, g = normalize(pred), normalize(gt)
    return editdistance.eval(p, g) / max(len(g), 1)


def cer(pred: str, gt: str) -> float:
    """Character Error Rate — lower is better."""
    return ned(pred, gt)


def wer(pred: str, gt: str) -> float:
    """Word Error Rate — lower is better."""
    import editdistance
    p_words = normalize(pred).split()
    g_words = normalize(gt).split()
    return editdistance.eval(p_words, g_words) / max(len(g_words), 1)


def bleu(pred: str, gt: str, max_n: int = 4) -> float:
    """Corpus BLEU-4 approximation for a single pair."""
    from collections import Counter
    import math

    p_tokens = normalize(pred).split()
    g_tokens = normalize(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        p_ngrams = Counter(tuple(p_tokens[i:i+n]) for i in range(len(p_tokens)-n+1))
        g_ngrams = Counter(tuple(g_tokens[i:i+n]) for i in range(len(g_tokens)-n+1))
        clipped = sum(min(c, g_ngrams[ng]) for ng, c in p_ngrams.items())
        total = sum(p_ngrams.values())
        scores.append(clipped / max(total, 1))

    if any(s == 0 for s in scores):
        return 0.0

    bp = min(1.0, math.exp(1 - len(g_tokens) / max(len(p_tokens), 1)))
    return bp * math.exp(sum(math.log(s) for s in scores) / max_n)


def token_f1(pred: str, gt: str) -> tuple[float, float, float]:
    """Token-level Precision, Recall, F1."""
    from collections import Counter
    p_tokens = Counter(normalize(pred).split())
    g_tokens = Counter(normalize(gt).split())
    common = sum((p_tokens & g_tokens).values())
    precision = common / max(sum(p_tokens.values()), 1)
    recall = common / max(sum(g_tokens.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def compute_all(pred: str, gt: str) -> dict:
    _, _, f1 = token_f1(pred, gt)
    return {
        "ned":  round(ned(pred, gt), 4),
        "cer":  round(cer(pred, gt), 4),
        "wer":  round(wer(pred, gt), 4),
        "bleu": round(bleu(pred, gt), 4),
        "f1":   round(f1, 4),
    }
