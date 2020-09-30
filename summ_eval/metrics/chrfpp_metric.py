# pylint: disable=C0103
from multiprocessing import Pool
import gin
import sacrebleu
from .metric import Metric

@gin.configurable
class ChrfppMetric(Metric):
    def __init__(self, ncorder=6, beta=2, n_workers=24):
        """
        Chrf++ metric
        Wrapper around sacrebleu: https://github.com/mjpost/sacrebleu

        Args:
                :param ncorder: character n-gram order
                :param beta: beta parameter to balance precision and recall
                :param n_workers: number of processes to use if using multiprocessing

        """
        self.ncorder = ncorder
        self.beta = beta
        self.n_workers = n_workers

    def evaluate_example(self, summary, reference):
        score = sacrebleu.sentence_chrf(summary, reference, order=self.ncorder, beta=self.beta)
        score_dict = {"chrf": score.score}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        references = [references]
        if aggregate:
            score = sacrebleu.corpus_chrf(summaries, references, order=self.ncorder, beta=self.beta)
            score_dict = {"chrf": score.score}
            return score_dict
        else:
            with Pool(processes=self.n_workers) as pool:
                results = pool.starmap(self.evaluate_example, zip(summaries, references))
            return results

    @property
    def supports_multi_ref(self):
        return True
