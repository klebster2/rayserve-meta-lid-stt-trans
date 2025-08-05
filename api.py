"""
This is generic API for Language Identification ~4017 languages, Speech-to-Text ~1162 languages, and Translation for ~200 languages.

Code: Creative Commons CC0-1.0 license
Models: CC-BY-NC-4.0 license (Meta)

Language Identification component comes from a quantized version of MMS-LID (4017 languages).
Transcription uses Wav2Vec2 with the Facebook MMS-1B-ALL model. (1162 languages)
Translation uses the NLLB-200-distilled-600M model. (200 languages)
"""

import abc
import json
import os
from pathlib import Path
from typing import Any, Dict

import ffmpeg
import httpx
import langcodes
import numpy as np
import onnxruntime as ort
import ray
import torch
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from huggingface_hub import hf_hub_download
from magic import Magic
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import (AutoFeatureExtractor, AutoModelForSeq2SeqLM,
                          AutoProcessor, AutoTokenizer, Wav2Vec2ForCTC)

LID_ENDPOINT = "/audio/languageidentification"
TRANSCRIPTION_ENDPOINT = "/audio/transcription"
TRANSLATION_ENDPOINT = "/text/translation"
ROOT_ENDPOINT = "/"

router = APIRouter()


@router.get(ROOT_ENDPOINT)
async def root() -> RedirectResponse:
    """Redirect root to the automatically generated docs."""
    return RedirectResponse("/docs")


def load_audio(file_bytes: bytes, sampling_rate: int) -> np.ndarray:
    """
    Convert raw audio bytes into a normalized float32 NumPy array,
    re-sampled to a single channel at the specified sampling rate.
    """
    out, _ = (
        ffmpeg.input("pipe:", threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sampling_rate)
        .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file_bytes)
    )
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


class BaseDeployment(
    abc.ABC
):  # pylint: disable=too-few-public-methods,missing-class-docstring
    @property
    def name(self) -> str:
        """
        Return the class name as a default "deployment name".
        """
        return self.__class__.__name__


class BaseMMSDeployment(BaseDeployment):  # pylint: disable=too-few-public-methods
    """
    Common functionality for loading config, onnx, etc.
    """

    _sampling_rate = 16000

    def _get_label2id(self) -> Dict[str, str]:
        """
        Download and parse the label->id mapping for MMS-LID 4017.
        """
        config_url = (
            "https://huggingface.co/api/resolve-cache/models/facebook/mms-lid-4017/6adb04d61a52e5989b0e65f0d59d4755d81a94e3/config.json"
        )
        if not Path("id2label.json").exists():
            with httpx.Client() as client, open("id2label.json", "wb") as file:
                response = client.get(config_url)
                file.write(response.content)

        with open("id2label.json", "r", encoding="utf-8") as file:
            return json.load(file)["id2label"]

    def _get_onnx_model_path(self, repo_id: str) -> Path:
        """
        Find or download the quantized ONNX model for the given huggingface repo.
        """
        local_model_path = self._get_model_path(repo_id) / Path(
            "onnx/model_quantized.onnx"
        )
        if not local_model_path.exists():
            downloaded = hf_hub_download(
                repo_id=repo_id, filename="onnx/model_quantized.onnx"
            )
            return Path(downloaded)
        return local_model_path

    def _get_model_path(self, model_id: str) -> Path:
        """
        Return the local path (HuggingFace caching location) for a model ID.
        """
        return (
            Path("~")
            / Path(".cache/huggingface/models")
            / Path(f"models--{model_id.replace('/', '--')}")
        )


@serve.deployment(
    ray_actor_options={"num_gpus": 0.1},
    autoscaling_config={
        "max_replicas": 1,
        "min_replicas": 0,
        "target_throughput": 1,
        "max_concurrent_queries": 1,
        "downscale_delay": 10, # aggressive downscale
    },
)
class LangIdDeployment(BaseMMSDeployment):
    """
    Runs Audio Language Identification using the Facebook MMS-LID model.
    """

    def __init__(self) -> None:
        onnx_repo_id: str = "Xenova/mms-lid-4017"
        self._processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
        cuda_exe_provider = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": torch.cuda.current_device(),
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                },
            )
        ]
        self._onnx_model = ort.InferenceSession(
            str(self._get_onnx_model_path(onnx_repo_id)),  # ensure string path
            providers=cuda_exe_provider,
        )
        self._id2label = self._get_label2id()
        self._sample_rate = 16000

    async def analyze_audio(self, audio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Language Identification on the input audio file.
        Cuts audio to 30s if longer.
        """
        audio_array = load_audio(audio["content"], self._sample_rate)
        if audio_array.shape[0] > self._sample_rate * 30:
            print("30 second limit exceeded, truncating audio")
            print("Change in code if this behaviour is not desired")
            audio_array = audio_array[: self._sample_rate * 30]

        inputs = self._processor(
            audio_array, sampling_rate=self._sample_rate, return_tensors="np"
        )
        input_name = self._onnx_model.get_inputs()[0].name
        input_array = inputs["input_values"].astype(np.float32)

        outputs = self._onnx_model.run(
            None,
            {input_name: input_array},
        )
        logits = outputs[0]
        best_id = np.argmax(logits, axis=-1)[0]
        detected_lang = self._id2label[str(best_id)]
        standardized_lang = langcodes.standardize_tag(detected_lang)
        autonym = langcodes.Language.get(detected_lang).autonym()
        print(
            "Detected language:",
            detected_lang,
            "Standardized tag:",
            standardized_lang,
            "Autonym:",
            autonym,
        )

        return {
            "langid": detected_lang,
            "autonym": autonym,
        }


class AudioAnalysisResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Response model for the /audio endpoint. Contains the detected language code."""

    langid: str


@router.post(LID_ENDPOINT, response_model=AudioAnalysisResponse)
async def analyze_audio_endpoint(
    audio: UploadFile = File(...),
) -> AudioAnalysisResponse:
    """
    Endpoint that calls the LangIdDeployment to detect the language.
    """
    responder: DeploymentHandle = serve.get_deployment_handle(LangIdDeployment.name)

    audio_bytes = await audio.read()
    file_info = {
        "content": audio_bytes,
        "filename": audio.filename,
        "content_type": audio.content_type,
    }
    result: Dict[str, Any] = await responder.analyze_audio.remote(file_info)

    if "error" in result:
        raise ValueError("Error occurred during language detection")

    return AudioAnalysisResponse(**result)


@serve.deployment(
    ray_actor_options={"num_gpus": 0},  # TODO: use model.to("cuda") for GPU
    autoscaling_config={
        "max_replicas": 1,
        "min_replicas": 0,
        "target_throughput": 1,
        "max_concurrent_queries": 1,
        "downscale_delay": 10, # aggressive downscale
    },
)
class TranscriptionDeployment(BaseMMSDeployment):
    """
    Runs speech-to-text using the Facebook MMS-1B-ALL model.
    """

    def __init__(self) -> None:
        model_id = "facebook/mms-1b-all"
        self._processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_id
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=model_id
        )

    def transcribe_audio(self, audio: Dict[str, Any], language: str) -> Dict[str, Any]:
        """
        Transcribe the input audio file.
        """
        self._processor.tokenizer.set_target_lang(language)
        try:
            self._model.load_adapter(language)
        except KeyError:
            return {"error": f"No adapter found for '{language}'"}

        audio_array = load_audio(audio["content"], sampling_rate=self._sampling_rate)[
            : 30 * self._sampling_rate
        ]
        inputs = self._processor(
            audio_array, sampling_rate=self._sampling_rate, return_tensors="pt"
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        transcription = self._processor.decode(ids)

        return {"transcription": transcription}


class TranscriptionResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Response model for the /transcribe endpoint."""

    transcription: str
    langid: str


@router.post(TRANSCRIPTION_ENDPOINT, response_model=TranscriptionResponse)
async def transcribe_endpoint(
    audio: UploadFile = File(...), language: str = ""
) -> TranscriptionResponse:
    """
    Transcribe the input audio file. If no language is specified,
    it will automatically detect a language via the audio-language identification model deployment first.
    """
    if language == "":
        detected = await analyze_audio_endpoint(audio)
        language = detected.langid
        await audio.seek(0)

    responder: DeploymentHandle = serve.get_deployment_handle(
        TranscriptionDeployment.name
    )
    audio_bytes = await audio.read()
    file_info = {
        "content": audio_bytes,
        "filename": audio.filename,
        "content_type": audio.content_type,
    }

    result: Dict[str, Any] = await responder.transcribe_audio.remote(
        file_info, language
    )

    if "error" in result:
        raise ValueError(f"Error occurred during transcription: {result['error']}")

    result.update({"langid": language})

    return TranscriptionResponse(**result)


class TranslateRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Request body for the /translate endpoint."""

    text: str
    src_lang: str
    tgt_lang: str


class TranslateResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Response model for the /translate endpoint."""

    translated_text: str


@serve.deployment(
    ray_actor_options={"num_gpus": 0.2},  # At most 0.2 (4.8GB) on an RTX 3090 (24GB)
    autoscaling_config={
        "max_replicas": 1,
        "min_replicas": 0,
        "target_throughput": 1,
        "max_concurrent_queries": 1,
        "downscale_delay": 10, # aggressive downscale
    },
)
class NLLBDeployment(BaseDeployment):
    """
    Deployment for translation using the NLLB-200-distilled-600M model,
    loaded with device_map="auto" to leverage available GPU(s).
    """

    def __init__(self) -> None:
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto"
        )
        self.languages = tuple(
            tok for tok in self.tokenizer.all_special_tokens if "<" not in tok
        )

    async def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate from src_lang to tgt_lang using NLLB.
        This model ISO-639-3 codes like fra, eng, etc. plus scripts (e.g. Latn, Cyrl, Arab).
        If the script is omitted, the model will attempt to infer it automatically by searching
        for a matching three characters at the beginning of the language code.
        Languages for nllb-200-distilled-600M are in format ISO-639-3 + Script Code
        """
        # TODO: Try to use a heavily quantized version of NLLB-Moe
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        _matching_languages = tuple(  # Note this enables some flexibility
            filter(lambda x: x.startswith(tgt_lang), self.languages)
        )

        if not _matching_languages:
            raise ValueError(f"Target language '{tgt_lang}' not found in model")

        if len(_matching_languages) > 1:
            raise ValueError(
                (
                    f"Ambiguous target language '{tgt_lang}' -> "
                    f"'{''.join(_matching_languages)}' - consider including script code"
                )
            )
        assert len(_matching_languages) == 1
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            _matching_languages[0]
        )

        outputs = self.model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_length=512
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


@router.post(TRANSLATION_ENDPOINT, response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
    """
    Translate text from a source language to a target language via NLLB.
    """
    responder: DeploymentHandle = serve.get_deployment_handle(NLLBDeployment.name)
    translated_text: str = await responder.translate_text.remote(
        request.text, request.src_lang, request.tgt_lang
    )
    return TranslateResponse(translated_text=translated_text)


app = FastAPI()
app.include_router(router)


@serve.deployment(name="mm-conv-ai")
@serve.ingress(app)
class App:  # pylint: disable=too-few-public-methods
    """
    Ray Serve application, bundling all endpoints.
    """

    def __init__(
        self,
        audio_responder: DeploymentHandle,
        transcription_responder: DeploymentHandle,
        translation_responder: DeploymentHandle,
    ):
        self.audio_responder = audio_responder
        self.transcription_responder = transcription_responder
        self.translation_responder = translation_responder


def run_smoke_test_audio(file_url: str, endpoint: str, description: str = "") -> None:
    """
    Helper function to download a file, detect its MIME, and call the local endpoint.
    """
    print(f"\n--- SMOKE TEST: {description} ---")
    tmp_name = file_url.split("/")[-1]
    with open(tmp_name, "wb") as f:
        f.write(httpx.get(file_url, follow_redirects=True).content)

    mime_type = Magic(mime=True).from_file(tmp_name)
    with open(tmp_name, "rb") as f:
        files = {"audio": (tmp_name, f, mime_type)}
        headers = {"accept": "application/json"}
        resp = httpx.post(
            f"http://127.0.0.1:8000{endpoint}", headers=headers, files=files, timeout=90
        )
    os.system(f"rm {tmp_name}")
    print(f"MIME type: {mime_type}")
    print(f"Status code: {resp.status_code}")
    print(f"Response: {resp.text}")


def run_smoke_test_text(
    text: str, tgt_lang: str, endpoint: str, description: str = ""
) -> None:
    """
    Helper function to call the local endpoint with a text input.
    """
    print(f"\n--- SMOKE TEST: {description} ---")
    print("Translating text:", text)
    headers = {"accept": "application/json"}
    data = {"text": text, "src_lang": "eng", "tgt_lang": tgt_lang}
    resp = httpx.post(f"http://127.0.0.1:8000{endpoint}", headers=headers, json=data)
    print(f"Status code: {resp.status_code}")
    print(f"Response: {resp.text}")


if __name__ == "__main__":
    ray.init()

    # Deploy everything
    serve.run(
        App.bind(  # pylint: disable=no-value-for-parameter,no-member
            audio_responder=LangIdDeployment.bind(),  # pylint: disable=no-value-for-parameter,no-member
            transcription_responder=TranscriptionDeployment.bind(),  # pylint: disable=no-value-for-parameter,no-member
            translation_responder=NLLBDeployment.bind(),  # pylint: disable=no-value-for-parameter,no-member
        )
    )
    # Uncomment this block to start running smoke tests..
    # ## Test 1: Simple French "merci" (should detect 'fra')
    # run_smoke_test_audio(
    #     "https://upload.wikimedia.org/wikipedia/commons/f/fa/Nl-merci.ogg",
    #     LID_ENDPOINT,
    #     description="French 'merci' language ID",
    # )

    # ## Test 2: buriat (expected 'bxm')
    # run_smoke_test_audio(
    #     "https://archive.phonetics.ucla.edu/Language/BXM/bxm_word-list_1991_01.mp3",
    #     LID_ENDPOINT,
    #     description="Buriat Mongolia Language word list language ID = BXM",
    # )

    # ## Test 3: Gettysburg address (English) - Transcribe
    # run_smoke_test_audio(
    #     "https://www.cs.uic.edu/~troy/spring09/cs101/SoundFiles/gettysburg.wav",
    #     TRANSCRIPTION_ENDPOINT,
    #     description="Gettysburg address transcription",
    # )

    # Test 4: French - LangId and Transcribe of 'Un Bleu'
    run_smoke_test_audio(
        "https://upload.wikimedia.org/wikipedia/commons/6/63/Fr-bleu.ogg",
        TRANSCRIPTION_ENDPOINT,
        description="French language transcription",
    )

    # # Test 5: English to French translation
    # run_smoke_test_text(
    #     "Hello, my name is John. I am a software Engineer from the United States.",
    #     "fra",
    #     TRANSLATION_ENDPOINT,
    #     description="English text translation",
    # )

    # print("\nReady to accept requests...\n")
    # while True:
    #     pass
