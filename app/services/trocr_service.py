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
        # Use every physical core; the previous set_num_threads(get_num_threads())
        # call was a no-op.
        torch.set_num_threads(max(1, os.cpu_count() or 1))
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

        Vectorized: one softmax + gather over the whole batch instead of a
        Python loop with a full-vocab softmax per token.
        """
        if not output.scores:
            return [0.0] * output.sequences.shape[0]

        # sequences includes the initial decoder_start token; scores align with
        # the generated tokens that follow it.
        generated = output.sequences[:, 1 : 1 + len(output.scores)]
        # (batch, steps, vocab)
        scores = torch.stack(list(output.scores), dim=1)
        steps = min(generated.shape[1], scores.shape[1])
        generated = generated[:, :steps]
        scores = scores[:, :steps]

        probs = torch.softmax(scores, dim=-1)
        token_probs = probs.gather(-1, generated.unsqueeze(-1)).squeeze(-1)

        pad_id = self._model.config.pad_token_id
        mask = torch.ones_like(token_probs, dtype=torch.bool)
        if pad_id is not None:
            mask = generated != pad_id

        confidences: list[float] = []
        for i in range(token_probs.shape[0]):
            valid = token_probs[i][mask[i]]
            confidences.append(
                round(float(valid.mean()), 4) if valid.numel() else 0.0
            )
        return confidences
