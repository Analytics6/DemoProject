import math
from collections import Counter
from typing import Dict, List, Sequence, Set


def _tokenize(text: str) -> List[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token.strip()]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


class TextMetrics:
    """Classic text-generation metrics (lightweight implementations)."""

    def bleu(self, reference: str, candidate: str, n_gram: int = 2) -> float:
        ref_tokens = _tokenize(reference)
        cand_tokens = _tokenize(candidate)
        if not ref_tokens or not cand_tokens:
            return 0.0

        def ngrams(tokens: List[str], n: int) -> Counter:
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        precision_scores: List[float] = []
        for n in range(1, n_gram + 1):
            ref_counts = ngrams(ref_tokens, n)
            cand_counts = ngrams(cand_tokens, n)
            overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
            total = sum(cand_counts.values())
            precision_scores.append(_safe_div(overlap, total))

        if any(score == 0 for score in precision_scores):
            return 0.0

        log_avg_precision = sum(math.log(score) for score in precision_scores) / len(precision_scores)
        brevity_penalty = (
            1.0
            if len(cand_tokens) > len(ref_tokens)
            else math.exp(1 - _safe_div(len(ref_tokens), len(cand_tokens)))
        )
        return brevity_penalty * math.exp(log_avg_precision)

    def rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        ref = _tokenize(reference)
        cand = _tokenize(candidate)
        ref_counts = Counter(ref)
        cand_counts = Counter(cand)
        overlap = sum(min(ref_counts[token], cand_counts[token]) for token in set(ref_counts))
        precision = _safe_div(overlap, len(cand))
        recall = _safe_div(overlap, len(ref))
        f1 = _safe_div(2 * precision * recall, precision + recall)
        return {"rouge_precision": precision, "rouge_recall": recall, "rouge_f1": f1}

    def meteor(self, reference: str, candidate: str) -> float:
        ref_set = set(_tokenize(reference))
        cand_set = set(_tokenize(candidate))
        overlap = len(ref_set & cand_set)
        precision = _safe_div(overlap, len(cand_set))
        recall = _safe_div(overlap, len(ref_set))
        return _safe_div(10 * precision * recall, recall + 9 * precision)

    def perplexity(self, token_probabilities: Sequence[float]) -> float:
        if not token_probabilities:
            return float("inf")
        clamped = [max(prob, 1e-12) for prob in token_probabilities]
        avg_neg_log = -sum(math.log(prob) for prob in clamped) / len(clamped)
        return math.exp(avg_neg_log)


class RagMetrics:
    """RAG-specific retrieval grounding quality metrics."""

    def context_precision(self, retrieved_context_ids: Set[str], relevant_context_ids: Set[str]) -> float:
        return _safe_div(len(retrieved_context_ids & relevant_context_ids), len(retrieved_context_ids))

    def context_recall(self, retrieved_context_ids: Set[str], relevant_context_ids: Set[str]) -> float:
        return _safe_div(len(retrieved_context_ids & relevant_context_ids), len(relevant_context_ids))

    def precision_at_k(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
        top_k = list(retrieved_ids[:k])
        hit = sum(1 for item in top_k if item in relevant_ids)
        return _safe_div(hit, len(top_k))

    def recall_at_k(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
        top_k = list(retrieved_ids[:k])
        hit = sum(1 for item in top_k if item in relevant_ids)
        return _safe_div(hit, len(relevant_ids))


class RetrievalMetrics:
    """Pure retrieval metrics (precision/recall/hit-rate/MRR/MAP)."""

    def precision_at_k(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
        top_k = list(retrieved_ids[:k])
        true_pos = sum(1 for item in top_k if item in relevant_ids)
        return _safe_div(true_pos, len(top_k))

    def recall_at_k(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
        top_k = list(retrieved_ids[:k])
        true_pos = sum(1 for item in top_k if item in relevant_ids)
        return _safe_div(true_pos, len(relevant_ids))

    def hit_rate_at_k(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
        top_k = set(retrieved_ids[:k])
        return 1.0 if top_k & relevant_ids else 0.0

    def mrr(self, retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
        for idx, item in enumerate(retrieved_ids, start=1):
            if item in relevant_ids:
                return 1.0 / idx
        return 0.0

    def map_score(self, retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 10) -> float:
        top_k = list(retrieved_ids[:k])
        running_hits = 0
        precision_sum = 0.0
        for idx, item in enumerate(top_k, start=1):
            if item in relevant_ids:
                running_hits += 1
                precision_sum += running_hits / idx
        return _safe_div(precision_sum, len(relevant_ids))


class GenerationMetrics:
    """Generation evaluation for quality and hallucination checks."""

    def faithfulness(self, generated_answer: str, retrieved_context: str) -> float:
        answer_tokens = set(_tokenize(generated_answer))
        context_tokens = set(_tokenize(retrieved_context))
        return _safe_div(len(answer_tokens & context_tokens), len(answer_tokens))

    def answer_relevance(self, question: str, generated_answer: str) -> float:
        q_tokens = set(_tokenize(question))
        a_tokens = set(_tokenize(generated_answer))
        return _safe_div(len(q_tokens & a_tokens), len(q_tokens))

    def answer_correctness(self, generated_answer: str, expected_answer: str) -> float:
        return TextMetrics().meteor(expected_answer, generated_answer)

    def hallucination(self, generated_answer: str, retrieved_context: str) -> float:
        # Fraction of answer tokens not supported by context.
        answer_tokens = set(_tokenize(generated_answer))
        context_tokens = set(_tokenize(retrieved_context))
        unsupported = answer_tokens - context_tokens
        return _safe_div(len(unsupported), len(answer_tokens))


class HumanInTheLoop:
    """Collects human feedback scores for continuous quality checks."""

    def review(self, helpfulness: int, correctness: int, safety: int, notes: str = "") -> Dict:
        bounded = lambda value: max(1, min(5, int(value)))
        result = {
            "helpfulness": bounded(helpfulness),
            "correctness": bounded(correctness),
            "safety": bounded(safety),
            "notes": notes.strip(),
        }
        result["overall"] = round(
            (result["helpfulness"] + result["correctness"] + result["safety"]) / 3.0, 2
        )
        return result
