import spacy

import gin
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from summ_eval.config import STATIC_PATH

from .metrics import (BertScoreMetric, BleuMetric, ChrfppMetric, CiderMetric,
                      MeteorMetric, MoverScoreMetric, RougeMetric,
                      RougeWeMetric, SentenceMoversMetric, SyntacticMetric)

gin.parse_config_file(STATIC_PATH / "basic.config")

all_metrics = {
    "bert_score": {"cls": BertScoreMetric, "toks_needed": {"space"}},
    "bleu": {"cls": BleuMetric, "toks_needed": {"space"}},
    "chrf": {"cls": ChrfppMetric, "toks_needed": {"space"}},
    "cider": {"cls": CiderMetric, "toks_needed": {"stem"}},
    "meteor": {"cls": MeteorMetric, "toks_needed": {"space"}},
    "mover_score": {"cls": MoverScoreMetric, "toks_needed": {"space"}},
    "rouge": {"cls": RougeMetric, "toks_needed": {"line_delimited"}},
    "rouge_we": {"cls": RougeWeMetric, "toks_needed": {"stem"}},
    "sms": {"cls": SentenceMoversMetric, "toks_needed": {"spacy"}},
    "syntactic": {"cls": SyntacticMetric, "toks_needed": {"space"}},
}


def extract_field(dict_list, field):
    processed = [entry[field] for entry in dict_list]
    processed = [[entry] if isinstance(entry, str) else entry for entry in processed]
    if any(not isinstance(entry, list) for entry in processed):
        raise ValueError("expected lists")
    return processed


def compute(metrics, summ_refs, aggregate=True):
    toks_needed = {
        tok for metric in metrics for tok in all_metrics[metric]["toks_needed"]
    }

    summaries = extract_field(summ_refs, "summary")
    references = extract_field(summ_refs, "reference")

    if "line_delimited" in toks_needed:
        references_delimited = ["\n".join(ref) for ref in references]
        summaries_delimited = ["\n".join(summ) for summ in summaries]

    if "space" in toks_needed:
        references_space = [" ".join(ref) for ref in references]
        summaries_space = [" ".join(summ) for summ in summaries]

    if "stem" in toks_needed:
        tokenizer = RegexpTokenizer(r"\w+")
        stemmer = SnowballStemmer("english")

        summaries_stemmed = [
            " ".join(stemmer.stem(word) for word in tokenizer.tokenize(" ".join(summ)))
            for summ in summaries
        ]
        references_stemmed = [
            " ".join(stemmer.stem(word) for word in tokenizer.tokenize(" ".join(ref)))
            for ref in references
        ]

    if "spacy" in toks_needed:
        nlp = spacy.load("en_core_web_md")
        disable = ["tagger", "textcat", "ner"]
        summaries_spacy = [nlp(" ".join(text), disable=disable) for text in summaries]
        if "sms" in metrics:
            references_spacy = [
                nlp(" ".join(text), disable=disable) for text in references
            ]

    final_output = {} if aggregate else []

    for metric in metrics:
        print(f"Calculating scores for {metric}.")
        metric_obj = all_metrics[metric]["cls"]()
        metrics_toks_needed = all_metrics[metric]["toks_needed"]

        if "line_delimited" in metrics_toks_needed:
            output = metric_obj .evaluate_batch(
                summaries_delimited, references_delimited, aggregate=aggregate
            )
        elif "space" in metrics_toks_needed:
            output = metric_obj .evaluate_batch(
                summaries_space, references_space, aggregate=aggregate
            )
        elif "stem" in metrics_toks_needed:
            output = metric_obj .evaluate_batch(
                summaries_stemmed, references_stemmed, aggregate=aggregate
            )
        elif metric == "sms":
            output = metric_obj .evaluate_batch(
                summaries_spacy, references_spacy, aggregate=aggregate
            )

        if aggregate:
            final_output.update(output)
        else:
            final_output.append(output)
        if isinstance(metric_obj, MeteorMetric):
            metric_obj.close()
        del metric_obj

    return final_output
