"""Handwritten text recognition using TrOCR.

The model is loaded exactly once (at application startup) and kept in memory
on app.state — never per request.
"""

import os

import numpy as np
import torch
from PIL import Image

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_CPU_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_CPU_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _read_first_line(path: str) -> str:
    with open(path) as handle:
        return handle.readline().strip()


def _available_cpus() -> int:
    """Cores this process may actually use.

    os.cpu_count() reports the host's cores, not the container's cgroup CPU
    quota — on a shared PaaS host that is a wildly inflated number (48 on
    Railway), and every torch thread carries its own workspace buffers. Read
    the cgroup v2/v1 quota first and fall back to the affinity mask.

    An unlimited quota ("max" on v2, -1 on v1) means no container limit, so
    the affinity mask is the right answer there.
    """
    # v2: single line "<quota|max> <period>". v1: quota and period in
    # separate files, quota -1 when unlimited.
    for quota_path, period_path in (
        (CGROUP_V2_CPU_MAX, None),
        (CGROUP_V1_CPU_QUOTA, CGROUP_V1_CPU_PERIOD),
    ):
        try:
            fields = _read_first_line(quota_path).split()
            if fields[0] == "max":
                break
            period = fields[1] if period_path is None else _read_first_line(period_path)
            cores = int(fields[0]) / int(period)
            if cores > 0:
                return max(1, int(cores))
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            continue

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on Windows/macOS
        return max(1, os.cpu_count() or 1)


class TrOCRService:
    """Batched TrOCR inference, optimized for CPU."""

    def __init__(self, settings: Settings) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Loading TrOCR model '%s'...", settings.trocr_model_name)
        self._device = torch.device(settings.device)
        self._processor = TrOCRProcessor.from_pretrained(
            settings.trocr_model_name, use_fast=True
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model_name)
        self._model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        if settings.quantize_trocr and self._device.type == "cpu":
            logger.info("Applying dynamic int8 quantization to TrOCR...")
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self._batch_size = settings.trocr_batch_size
        self._max_new_tokens = settings.trocr_max_new_tokens
        torch.set_num_threads(_available_cpus())
        logger.info(
            "TrOCR model ready on %s (%d torch threads, quantized=%s).",
            self._device,
            torch.get_num_threads(),
            settings.quantize_trocr and self._device.type == "cpu",
        )

    def recognize(self, line_images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize a list of cropped line images.

        Returns one (text, confidence) tuple per input, in the same order.
        Confidence is the exponential of the mean token log-probability.
        """
        results: list[tuple[str, float]] = []
        for start in range(0, len(line_images), self._batch_size):
            batch = line_images[start : start + self._batch_size]
            results.extend(self._recognize_batch(batch))
        return results

    def _recognize_batch(self, batch: list[np.ndarray]) -> list[tuple[str, float]]:
        pil_images = [Image.fromarray(img).convert("RGB") for img in batch]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self._device)

        with torch.inference_mode():
            output = self._model.generate(
                pixel_values,
                max_new_tokens=self._max_new_tokens,
                num_beams=1,  # greedy — guaranteed, regardless of the model's generation_config
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self._processor.batch_decode(output.sequences, skip_special_tokens=True)
        confidences = self._sequence_confidences(output)
        return [(text.strip(), conf) for text, conf in zip(texts, confidences)]

    def _sequence_confidences(self, output) -> list[float]:
        """Average per-token probability of the chosen tokens for each sequence.

        Softmax and gather run one decoding step at a time. Stacking every
        step into a single (batch, steps, vocab) tensor and softmaxing that
        allocates two more copies of the full logits — with TrOCR's 50k vocab
        that is hundreds of MB per request, which OOM-killed the container on
        receipts with many lines. Per step the temporary is (batch, vocab) and
        is freed on the next iteration.
        """
        if not output.scores:
            return [0.0] * output.sequences.shape[0]

        # sequences includes the initial decoder_start token; scores align with
        # the generated tokens that follow it.
        generated = output.sequences[:, 1 : 1 + len(output.scores)]
        steps = min(generated.shape[1], len(output.scores))
        if steps == 0:
            return [0.0] * output.sequences.shape[0]

        pad_id = self._model.config.pad_token_id

        totals = torch.zeros(generated.shape[0], dtype=torch.float32)
        counts = torch.zeros(generated.shape[0], dtype=torch.float32)
        for step in range(steps):
            tokens = generated[:, step]
            step_probs = torch.softmax(output.scores[step].float(), dim=-1)
            token_probs = step_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            keep = (
                torch.ones_like(token_probs, dtype=torch.bool)
                if pad_id is None
                else tokens != pad_id
            )
            totals += torch.where(keep, token_probs, torch.zeros_like(token_probs))
            counts += keep.to(totals.dtype)

        return [
            round(float(total / count), 4) if count else 0.0
            for total, count in zip(totals.tolist(), counts.tolist())
        ]
