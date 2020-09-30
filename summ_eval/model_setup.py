import bz2
import inspect
import io
import logging
import zipfile
from math import ceil
from pathlib import Path
from threading import Thread

import requests
from bert_score import BERTScorer as Bert

import moverscore
import nltk
from summ_eval.config import CACHE_PATH


def download_file(
    url, save_path, logger=None, bz2_=False, zip_=False, chunk_size=20 * 1024 ** 2
):
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    with requests.get(url, stream=True) as req:
        req.raise_for_status()

        if zip_:
            with zipfile.ZipFile(io.BytesIO(req.content)) as file:
                file.extractall(save_path)
            return

        length = req.headers.get("Content-length")
        total = ceil(int(length) / chunk_size) if length is not None else None
        chunks = req.iter_content(chunk_size=chunk_size)

        if bz2_:
            bz2_decompressor = bz2.BZ2Decompressor()

        with open(save_path, "wb") as file:
            for i, chunk in enumerate(chunks, start=1):
                if bz2_:
                    chunk = bz2_decompressor.decompress(chunk)
                file.write(chunk)
                if logger is not None:
                    logger.info(f"{i}/{total}")


def setup_meteor():
    meteor_url = "https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar?raw=true"
    paragraph_url = "https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/data/paraphrase-en.gz?raw=true"
    meteor_path = CACHE_PATH / "meteor-1.5.jar"
    paragraph_path = CACHE_PATH / "data" / "paraphrase-en.gz"

    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    if not meteor_path.exists():
        logger.info("downloading")
        download_file(meteor_url, meteor_path, logger=logger)
    else:
        logger.info("already present")

    if not paragraph_path.exists():
        logger.info("downloading")
        download_file(paragraph_url, paragraph_path, logger=logger)
    else:
        logger.info("already present")
    logger.info("done")


def setup_s3_rouge_we():
    url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
    embeddings_path = CACHE_PATH / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)

    deps_words_path = embeddings_path / "deps.words"

    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    if not deps_words_path.exists():
        logger.info("downloading")
        download_file(url, deps_words_path, logger=logger, bz2_=True)
        logger.info("done")
    else:
        logger.info("already present")


def setup_corenlp_syntactic():
    url = "http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
    corenlp_path = CACHE_PATH / "stanford-corenlp-full-2018-10-05"

    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    if not corenlp_path.exists():
        logger.info("downloading")
        download_file(url, CACHE_PATH, zip_=True)
        logger.info("done")
    else:
        logger.info("already present")


def setup_nltk():
    logger = logging.getLogger(inspect.currentframe().f_code.co_name)

    logger.info("setup nltk")
    nltk.download("stopwords")
    logger.info("done")


def setup_bert():
    model_name = "bert-base-uncased"
    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    logger.setLevel(logging.INFO)
    logger.info("begin")

    try:
        logger.info("download model file...")
        Bert(model_type=model_name)
        logger.info("download model file done")
    except Exception as ex:
        logger.exception("problem downloading model file: %s", ex)
    logger.info("done")


def setup_moverscorev1():
    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    logger.setLevel(logging.INFO)
    logger.info("begin")
    moverscore.MoverScore.model_setup()
    logger.info("done")


def setup_moverscorev2():
    model_name = "bert-base-uncased"
    logger = logging.getLogger(inspect.currentframe().f_code.co_name)
    logger.setLevel(logging.INFO)
    logger.info("begin")
    moverscore.MoverScoreV2.model_setup()
    logger.info("done")


def setup():
    threads = []
    threads.append(Thread(target=setup_meteor, daemon=True))
    threads.append(Thread(target=setup_s3_rouge_we, daemon=True))
    threads.append(Thread(target=setup_corenlp_syntactic, daemon=True))
    threads.append(Thread(target=setup_nltk, daemon=True))
    threads.append(Thread(target=setup_bert, daemon=True))
    threads.append(Thread(target=setup_moverscorev1, daemon=True))
    threads.append(Thread(target=setup_moverscorev2, daemon=True))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    FORMAT = "{asctime} {levelname} [{name}] {message}"
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, style="{", level=logging.INFO)
    setup()
