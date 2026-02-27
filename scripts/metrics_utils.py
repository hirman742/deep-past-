from __future__ import annotations

import math
from typing import Any

import sacrebleu


def compute_translation_metrics(
    *,
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    bleu_metric = sacrebleu.metrics.BLEU()
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    bleu = float(bleu_metric.corpus_score(predictions, [references]).score)
    chrfpp = float(chrf_metric.corpus_score(predictions, [references]).score)
    bleu_01 = bleu / 100.0
    chrfpp_01 = chrfpp / 100.0
    geom = math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))
    geom_01 = math.sqrt(max(bleu_01, 0.0) * max(chrfpp_01, 0.0))
    return {
        "bleu": float(bleu),
        "chrfpp": float(chrfpp),
        "geom": float(geom),
        "bleu_01": float(bleu_01),
        "chrfpp_01": float(chrfpp_01),
        "geom_01": float(geom_01),
    }


def build_metric_signatures() -> dict[str, Any]:
    bleu_metric = sacrebleu.metrics.BLEU()
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    probe_predictions = ["x"]
    probe_references = [["x"]]
    bleu_metric.corpus_score(probe_predictions, probe_references)
    chrf_metric.corpus_score(probe_predictions, probe_references)
    return {
        "sacrebleu_version": str(sacrebleu.__version__),
        "bleu": str(bleu_metric.get_signature()),
        "chrfpp": str(chrf_metric.get_signature()),
    }
