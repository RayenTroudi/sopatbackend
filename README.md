# SOPAT Handwriting OCR Backend

FastAPI service that extracts **handwritten text** from images uploaded by the SOPAT Flutter app.

**Pipeline:** upload → validate → EXIF orientation fix → resize/deskew/denoise/adaptive-threshold → **PaddleOCR** (text-line detection only) → crop each line → **TrOCR** (`microsoft/trocr-base-handwritten`, batched recognition) → merge lines in reading order → JSON.

Models load **once at startup** and stay in memory (`app.state`). Inference runs with `torch.inference_mode()` on CPU, in a worker thread so the event loop stays responsive.

## Project structure

```
app/
  main.py                 # FastAPI app, lifespan model loading, CORS
  config.py               # env-driven settings (pydantic-settings)
  api/
    routes.py             # /health, /ocr endpoints
    schemas.py            # response models
  services/
    paddle_service.py     # PaddleOCR detection + reading-order sort
    trocr_service.py      # TrOCR batched recognition + confidence
    image_processing.py   # decode, orientation, deskew, enhance, crop
    ocr_pipeline.py       # orchestration
  utils/logger.py
tests/                    # unit tests (run without models)
requirements.txt
Dockerfile
```

## Installation (local)

Requires **Python 3.11+** (the Docker image uses 3.11).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# CPU-only torch (smaller download), then the rest:
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust if needed. All settings are environment variables — see that file for the full list (model name, upload size, host, port, CORS, batch size...).

## Running locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First startup downloads the TrOCR (~1.3 GB) and PaddleOCR detection models; subsequent starts use the local cache. Swagger UI: **http://localhost:8000/docs**

## Docker

```bash
docker build -t sopat-ocr .
docker run -p 8000:8000 sopat-ocr
```

Models are baked into the image at build time, so the container starts fast and works offline.

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /ocr` — multipart/form-data, field `image`

```bash
curl -X POST http://localhost:8000/ocr -F "image=@note.jpg"
```

Success (200):

```json
{
  "success": true,
  "text": "Meeting notes\ncheck valve pressure",
  "confidence": 0.87,
  "lines": [
    { "text": "Meeting notes", "confidence": 0.91 },
    { "text": "check valve pressure", "confidence": 0.83 }
  ],
  "processing_time": 1.42,
  "request_id": "a1b2c3d4e5f6"
}
```

Failure:

```json
{ "success": false, "error": "Unable to detect handwriting.", "request_id": "..." }
```

| Status | Meaning |
|--------|---------|
| 400 | Invalid/corrupted image |
| 413 | File exceeds `MAX_UPLOAD_SIZE_MB` |
| 415 | Unsupported content type |
| 422 | Empty upload, missing `image` field, or no handwriting detected |
| 500 | Internal OCR failure |

## Flutter integration

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<String?> extractHandwriting(String imagePath) async {
  final request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_SERVER:8000/ocr'),
  );
  request.files.add(await http.MultipartFile.fromPath('image', imagePath));

  final response = await http.Response.fromStream(await request.send());
  final body = jsonDecode(response.body) as Map<String, dynamic>;

  if (response.statusCode == 200 && body['success'] == true) {
    return body['text'] as String;
  }
  print('OCR failed: ${body['error']}');
  return null;
}
```

Tip: compress/resize images client-side (e.g. `flutter_image_compress`, longest side ≤ 2000 px, JPEG quality ~85) to cut upload time — the server resizes anyway.

## Tests

The unit tests mock the models, so they run in seconds without downloading anything:

```bash
pytest tests/ -v
```

## Performance notes

- CPU inference of TrOCR-base takes roughly 0.5–2 s per line; lines are batched (`TROCR_BATCH_SIZE`).
- For higher accuracy at higher latency, set `TROCR_MODEL_NAME=microsoft/trocr-large-handwritten`.
- Detection runs on an enhanced (denoised + adaptive-thresholded) copy of the image, while TrOCR receives crops from the original image — TrOCR performs better on natural strokes than on hard-binarized ones.
