"""Shallow fusion via N-best rescoring with a KenLM language model.

CTranslate2's Whisper model does not expose step-by-step decoding, so we
cannot inject LM scores during beam search.  Instead we:

  1. Generate N hypotheses per sample via beam search (num_hypotheses > 1).
  2. Rescore each hypothesis:  score = acoustic + lm_weight * lm_score.
  3. Select the hypothesis with the highest combined score.
"""

import math
from typing import List

from whisperx.log_utils import get_logger

logger = get_logger(__name__)

_LN10 = math.log(10)


class KenLMFusion:
    """N-best rescoring with a KenLM language model.

    Parameters
    ----------
    model_path : str
        Path to a KenLM ``.arpa`` or ``.bin`` file.
    lm_weight : float
        Lambda weight for the LM score in the fusion formula.
    num_hypotheses : int
        Number of hypotheses to request from CTranslate2 beam search.
    """

    def __init__(
        self,
        model_path: str,
        lm_weight: float = 0.1,
        num_hypotheses: int = 5,
    ):
        try:
            import kenlm
        except ImportError:
            raise ImportError(
                "kenlm is required for LM fusion. Install with: "
                "pip install https://github.com/kpu/kenlm/archive/master.zip"
            )

        self.model = kenlm.Model(model_path)
        self.lm_weight = lm_weight
        self.num_hypotheses = num_hypotheses

        logger.info(
            "Loaded KenLM model from %s (order=%d, lm_weight=%.2f, "
            "num_hypotheses=%d)",
            model_path,
            self.model.order,
            lm_weight,
            num_hypotheses,
        )

    def score_text(self, text: str) -> float:
        """Return the per-word KenLM log-probability in natural log.

        The total sentence log-prob is divided by word count so that
        hypotheses of different lengths are comparable (avoids brevity
        bias where shorter hypotheses always win).
        """
        total_log10 = self.model.score(text, bos=True, eos=True)
        n_words = max(1, len(text.split()))
        return (total_log10 / n_words) * _LN10

    def rescore(
        self,
        hypotheses: List[str],
        acoustic_scores: List[float],
    ) -> int:
        """Rescore N-best hypotheses and return the index of the best.

        Parameters
        ----------
        hypotheses : list[str]
            Decoded text for each hypothesis.
        acoustic_scores : list[float]
            Average log-probability (natural log) from Whisper for each
            hypothesis.

        Returns
        -------
        int
            Index of the best hypothesis after rescoring.
        """
        if len(hypotheses) <= 1:
            return 0

        best_idx = 0
        best_score = float("-inf")

        for i, (text, acoustic_score) in enumerate(
            zip(hypotheses, acoustic_scores)
        ):
            lm_score = self.score_text(text.strip())
            combined = acoustic_score + self.lm_weight * lm_score

            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx != 0:
            logger.debug(
                "LM rescoring changed selection: idx=%d, "
                "acoustic=%.3f, combined=%.3f, text=%r",
                best_idx,
                acoustic_scores[best_idx],
                best_score,
                hypotheses[best_idx][:80],
            )

        return best_idx
