"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable values. Override any field via environment variable
    (e.g. TROCR_MODEL_NAME, MAX_UPLOAD_SIZE_MB, HOST, PORT)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # CORS
    cors_allow_origins: str = "*"  # comma-separated list, "*" for all

    # Upload constraints
    max_upload_size_mb: int = 10
    allowed_content_types: str = "image/jpeg,image/png,image/webp,image/bmp,image/tiff"

    # Models
    trocr_model_name: str = "microsoft/trocr-base-handwritten"
    paddle_lang: str = "en"
    device: str = "cpu"

    # Inference
    trocr_batch_size: int = 8
    trocr_max_new_tokens: int = 96
    detection_min_confidence: float = 0.5
    # Dynamic int8 quantization of TrOCR linear layers. Benchmarked on this
    # model: destroys recognition quality (degenerate token loops, garbage
    # text) — keep OFF unless re-validated against a test set.
    quantize_trocr: bool = False
    # Per-region 180° angle classification. Off by default: the pipeline
    # already deskews the whole image, so this step only costs time.
    use_angle_cls: bool = False

    # Preprocessing
    max_image_dimension: int = 2500
    min_image_dimension: int = 300

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_types(self) -> set[str]:
        return {t.strip().lower() for t in self.allowed_content_types.split(",") if t.strip()}

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
