# Ray Serve Meta Multilingual LID / STT / Translation tools

**A Ray Serve-based microservice providing:**

1. **Language Identification** for ~4,017 languages (via a quantized version of [MMS-LID](https://huggingface.co/facebook/mms-lid-4017)).
2. **Speech-to-Text** for ~1,162 languages (via [MMS-1B-ALL](https://huggingface.co/facebook/mms-1b-all)).
3. **Translation** for ~200 languages (via [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)).

This repository is inspired by Meta’s [Massively Multilingual Speech (MMS)](https://ai.facebook.com/research/publications/massively-multilingual-speech) and [No Language Left Behind (NLLB)](https://ai.facebook.com/research/no-language-left-behind/) initiatives. By combining these open-source models, the goal is to surface language technologies for diverse and low-resource languages.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Service](#running-the-service)
  - [Endpoints](#endpoints)
  - [Example Requests](#example-requests)
- [Smoke Testing](#smoke-testing)
- [License](#license)
- [References and Acknowledgments](#references-and-acknowledgments)

---

## Features

1. **Language Identification**

   - Identifies the language of an audio clip.
   - Uses a quantized ONNX version of MMS-LID, supporting over 4,000 language IDs.
   - TODO: ensure CUDA execution on ONNX

2. **Speech-to-Text**

   - Transcribes audio into text, using [Facebook MMS-1B-ALL](https://huggingface.co/facebook/mms-1b-all).
   - Over 1,000 languages supported with the appropriate language adapter.
   - TODO: ensure CUDA execution on ONNX

3. **Text Translation**

   - Translates text between 200 languages using [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
   - Supports ISO 639-3 language codes with script codes (e.g., `fra_Latn`, `eng_Latn`, etc.).
   - TODO: Add ONNX implementation
   - TODO: ensure CUDA execution

4. **Ray Serve Microservice**
   - Provides a FastAPI-based interface served by Ray.
   - Automatic scaling of replicas (GPU or CPU usage can be specified).

---

## Architecture Overview

- The **LangIdDeployment** handles language identification (`mms-lid-4017`).
- The **TranscriptionDeployment** handles audio transcription (`mms-1b-all`).
- The **NLLBDeployment** handles text translation (`nllb-200-distilled-600M`).
- A single **App** deployment includes the FastAPI routes and ties them all together.

---

## Requirements

- **Python** 3.10 (recommended)
- **Conda** for installing dependencies
- **FFmpeg** (for audio processing)
- **GPU** is optional but recommended for faster inference
- **CUDA** >=12.4

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/klebster2/rayserve-meta-lid-stt-trans
   cd rayserve-meta-lid-stt-trans
   ```

2. **Set up a Python environment**:

   ```bash
   conda env create -f environment.yml
   conda activate rayserve-meta-lid-stt-trans
   ```

---

## Usage

### Running the Service

1. **Start the Ray cluster** in a separate terminal:

   ```bash
   ray start --head
   ```

   or

   ```bash
   ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=1
   ```

   Or simply let Ray automatically start in local mode when you run the script.

2. **Run the main script**:
   ```bash
   python api.py
   ```
   This will:
   - Initialize Ray
   - Deploy the `LangIdDeployment`, `TranscriptionDeployment`, and `NLLBDeployment`
   - Start a FastAPI server with the endpoints defined in `app = FastAPI()`
   - Print logs as it runs any built-in smoke tests (if configured)

**By default**, the service will be available at `http://127.0.0.1:8000`.
Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for an auto-generated Swagger UI.

### Endpoints

1. **Root** – `GET /`

   - Redirects to `/docs` for the Swagger UI.

2. **Language Identification** – `POST /audio/languageidentification`

   - Accepts an audio file (Form data).
   - Returns a JSON object with the detected language code (ISO 639-3) and the autonym.

3. **Speech-to-Text** – `POST /audio/transcription`

   - Accepts an audio file (Form data) and an optional `language` query parameter.
   - If `language` is not provided, it will first call the language identification endpoint to guess the language.
   - Returns the transcription and the language code used.

4. **Translation** – `POST /text/translation`
   - Accepts a JSON body with `{ "text": "...", "src_lang": "...", "tgt_lang": "..." }`.
   - Returns the translated text using the NLLB-200-distilled model.

### Example Requests

Using `curl` from the command line, here are some basic examples:

1. **Language Identification**:

   ```bash
   curl -X POST "http://127.0.0.1:8000/audio/languageidentification" \
        -H "accept: application/json" \
        -F "audio=@/path/to/your_audio.mp3"
   ```

2. **Transcription** (with automatic language detection):

   ```bash
   curl -X POST "http://127.0.0.1:8000/audio/transcription" \
        -H "accept: application/json" \
        -F "audio=@/path/to/your_audio.wav"
   ```

3. **Transcription** (specifying a language):

   ```bash
   curl -X POST "http://127.0.0.1:8000/audio/transcription?language=fra" \
        -H "accept: application/json" \
        -F "audio=@/path/to/french_audio.wav"
   ```

4. **Translation**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/text/translation" \
        -H "Content-Type: application/json" \
        -d '{"text":"Hello, world!", "src_lang":"eng", "tgt_lang":"fra"}'
   ```

---

## Smoke Testing

The script includes `run_smoke_test_audio` and `run_smoke_test_text` helper functions that download test files and call the local endpoints. When you run `python your_script.py`, it performs a few sample calls:

- French “merci” (should detect `fra`)
- Buriat sample (expected `bxm`)
- Gettysburg address (English) for transcription
- Yoruba audio sample for transcription
- Simple English-to-French translation

You can customize or remove these tests in the `__main__` section.

---

## License

- **Code**: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) – You’re free to use and adapt without restriction.
- **Models**: The pretrained models from Meta (MMS-LID, MMS-1B-ALL, and NLLB) are released under a [CC-BY-NC-4.0 license](https://github.com/facebookresearch/fairseq/blob/main/LICENSE).
  - Please consult each model’s license (see their Hugging Face model pages) for usage terms and attribution requirements.
  - **Important**: Commercial usage may be restricted under the CC-BY-NC-4.0 license.

---

## References and Acknowledgments

- [Meta AI: Massively Multilingual Speech (MMS)](https://ai.meta.com/blog/multilingual-model-speech-recognition/)
- [Meta AI: No Language Left Behind (NLLB)](https://ai.meta.com/research/no-language-left-behind/)
- Hugging Face model repositories:
  - [facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017)
  - [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)
  - [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for serving the FastAPI application at scale.
- [ISO 639-3 Language Codes](https://en.wikipedia.org/wiki/ISO_639-3)

### Also See:

- [Unicef - Why Mother Tongue Education holds the key to unlocking every child's potential ](https://www.unicef.org/india/stories/why-mother-tongue-education-holds-key-unlocking-every-childs-potential)
- [Letter to the UK Parliament By Lucy-Crompton Reid (Chief Executive, Wikimedia UK)](https://committees.parliament.uk/writtenevidence/119180/pdf/)
- [Unesco The world atlas of languages](https://en.wal.unesco.org/world-atlas-languages)
- [Unesco World Atlas of Languages - Summary Document](https://unesdoc.unesco.org/ark:/48223/pf0000380132/PDF/380132eng.pdf.multi)
- [Atlas of the World’s Languages in Danger](https://unesdoc.unesco.org/in/documentViewer.xhtml?v=2.1.196&id=p::usmarcdef_0000187026&file=/in/rest/annotationSVC/DownloadWatermarkedAttachment/attach_import_70c069f5-be69-478d-80ca-47a6ce68c154%3F_%3D187026eng.pdf&locale=en&multi=true&ark=/ark:/48223/pf0000187026/PDF/187026eng.pdf)
- [CIA world factbook - Languages](https://www.cia.gov/the-world-factbook/field/languages/)
- [UCLA Phonetics Lab Data](http://archive.phonetics.ucla.edu/main2.htm)
- [ISO 639-3 Criticism](https://en.wikipedia.org/wiki/ISO_639-3#Criticism)

If you use or build upon this repository, please consider citing or mentioning the original models and referencing Meta’s relevant research.

---

**Enjoy building massively multilingual conversational AI!**
