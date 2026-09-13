from __future__ import annotations
import copy
import random
from typing import Tuple, Union, List
import json
import re
from urllib import request as urllib_request
from urllib import error as urllib_error

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""


# Dictionary global per session_id
CHAT_SESSIONS: dict[str, ChatSession] = {}

# Function to filter enabled options
def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the ollama options whose 'enable_*' flag is True."""
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None

def openai_base_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def lmstudio_root_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/api/v1"):
        return base[:-7]
    if base.endswith("/v1"):
        return base[:-3]
    return base


def lmstudio_native_base_url(url: str) -> str:
    return f"{lmstudio_root_url(url)}/api/v1"


def extract_message_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def openai_extract_text(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    return extract_message_text(message.get("content", ""))


def lmstudio_extract_text(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    reasoning_content = extract_message_text(message.get("reasoning_content", ""))
    content = extract_message_text(message.get("content", ""))

    if reasoning_content and content:
        return f"<think>{reasoning_content}</think>\n\n{content}"
    if reasoning_content:
        return f"<think>{reasoning_content}</think>"
    return content


def openai_request(method: str, url: str, api_key: str = "", payload: dict | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib_request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib_request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib_error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {error_body}") from e
    except urllib_error.URLError as e:
        raise RuntimeError(f"Error calling {url}: {e.reason}") from e


def strip_thinking_blocks(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    cleaned = text
    leading_think_pattern = re.compile(r"^\s*<think\b[^>]*>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)

    while True:
        updated = leading_think_pattern.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated

    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()


class OllamaApiAdapter:
    def __init__(self, url: str, api_key: str = ""):
        self.url = url
        self.api_key = api_key
        self.client = Client(host=url)

    def list_models(self) -> List[str]:
        models = self.client.list().get("models", [])
        try:
            return [model["model"] for model in models]
        except Exception:
            return [model["name"] for model in models]

    def generate_image_question(
        self,
        model: str,
        system: str,
        prompt: str,
        images,
        keep_alive: int,
        format: str,
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        debug: str,
    ) -> str:
        used_keep_alive = str(keep_alive) + "m"
        if keep_alive == 0:
            used_keep_alive = 0

        options = {
            "seed": seed,
            "top_k": top_k,
            "min_p": min_p,
            "top_p": top_p,
            "temperature": temperature,
            "repeat_penalty": repetition_penalty,
        }
        if max_new_tokens == 0:
            options["num_predict"] = -1
        else:
            options["num_predict"] = max_new_tokens

        response = self.client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images,
            keep_alive=used_keep_alive,
            format=format,
            options=options,
        )
        return response.get("response", "")

    def maybe_unload_model(self, model: str, debug: str):
        unload_model(self.client, model)


class OpenAICompatibleApiAdapter:
    def __init__(self, url: str, api_key: str = ""):
        self.url = url
        self.api_key = api_key
        self.base_url = openai_base_url(url)

    def list_models(self) -> List[str]:
        response = openai_request(
            "GET",
            f"{self.base_url}/models",
            api_key=self.api_key,
        )
        models = response.get("data", [])
        return [model.get("id", "") for model in models if isinstance(model, dict) and model.get("id")]

    def build_payload(
        self,
        model: str,
        system: str,
        prompt: str,
        images,
        keep_alive: int,
        format: str,
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        debug: str,
    ) -> dict:
        user_content = [{"type": "text", "text": prompt}]
        if images:
            for image_binary in images:
                encoded_image = base64.b64encode(image_binary).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        },
                    }
                )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_new_tokens != 0:
            payload["max_tokens"] = max_new_tokens

        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        return payload

    def extract_response_text(self, response: dict) -> str:
        return openai_extract_text(response)

    def should_explicitly_unload(self, keep_alive: int, keep_model_loaded: bool) -> bool:
        return not keep_model_loaded

    def generate_image_question(
        self,
        model: str,
        system: str,
        prompt: str,
        images,
        keep_alive: int,
        format: str,
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        debug: str,
    ) -> str:
        if debug == "enable":
            print(
                "[OpenAI Compatible]\n"
                "Ignoring unsupported or provider-specific options when unavailable: "
                "keep_alive, min_p, top_k, repetition_penalty"
            )

        payload = self.build_payload(
            model=model,
            system=system,
            prompt=prompt,
            images=images,
            keep_alive=keep_alive,
            format=format,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            debug=debug,
        )

        try:
            response = openai_request(
                "POST",
                f"{self.base_url}/chat/completions",
                api_key=self.api_key,
                payload=payload,
            )
        except RuntimeError:
            if payload.get("response_format"):
                if debug == "enable":
                    print("[OpenAI Compatible] response_format unsupported, retrying without it")
                payload.pop("response_format", None)
                response = openai_request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    api_key=self.api_key,
                    payload=payload,
                )
            else:
                raise

        return self.extract_response_text(response)

    def maybe_unload_model(self, model: str, debug: str):
        if debug == "enable":
            print("[OpenAI Compatible] unload not supported by the OpenAI API contract, skipping")


class LMStudioApiAdapter(OpenAICompatibleApiAdapter):
    def __init__(self, url: str, api_key: str = ""):
        super().__init__(lmstudio_root_url(url), api_key)
        self.native_base_url = lmstudio_native_base_url(url)

    def list_models(self) -> List[str]:
        response = openai_request(
            "GET",
            f"{self.native_base_url}/models",
            api_key=self.api_key,
        )
        models = response.get("models") or response.get("data") or []
        return [
            model.get("key") or model.get("modelKey") or model.get("id", "")
            for model in models
            if isinstance(model, dict)
            and (model.get("type") in (None, "llm", "model") or model.get("object") == "model")
            and (model.get("key") or model.get("modelKey") or model.get("id"))
        ]

    def build_payload(
        self,
        model: str,
        system: str,
        prompt: str,
        images,
        keep_alive: int,
        format: str,
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        debug: str,
    ) -> dict:
        payload = super().build_payload(
            model=model,
            system=system,
            prompt=prompt,
            images=images,
            keep_alive=keep_alive,
            format=format,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            debug=debug,
        )

        # LM Studio supports OpenAI-compatible inference parameters, including top_k,
        # repeat_penalty, seed, and TTL.
        payload["top_k"] = top_k
        payload["repeat_penalty"] = repetition_penalty
        payload["seed"] = seed

        if keep_alive > 0:
            payload["ttl"] = keep_alive * 60

        return payload

    def extract_response_text(self, response: dict) -> str:
        return lmstudio_extract_text(response)

    def should_explicitly_unload(self, keep_alive: int, keep_model_loaded: bool) -> bool:
        return keep_alive == 0 or not keep_model_loaded

    def maybe_unload_model(self, model: str, debug: str):
        response = openai_request(
            "GET",
            f"{self.native_base_url}/models",
            api_key=self.api_key,
        )
        models = response.get("models") or response.get("data") or []

        selected_model = None
        for model_entry in models:
            model_key = model_entry.get("key") or model_entry.get("modelKey") or model_entry.get("id")
            if isinstance(model_entry, dict) and model_key == model:
                selected_model = model_entry
                break

        if not selected_model:
            if debug == "enable":
                print(f"[LM Studio] could not find model metadata for {model}, skipping unload")
            return

        loaded_instances = selected_model.get("loaded_instances") or selected_model.get("loadedInstances") or []
        if not loaded_instances:
            if debug == "enable":
                print(f"[LM Studio] model {model} is not currently loaded, skipping unload")
            return

        instance_id = loaded_instances[0].get("id") or loaded_instances[0].get("instance_id") or loaded_instances[0].get("instanceId")
        if not instance_id:
            if debug == "enable":
                print(f"[LM Studio] no instance id found for model {model}, skipping unload")
            return

        if debug == "enable":
            print(f"[LM Studio] unloading instance {instance_id}")

        openai_request(
            "POST",
            f"{self.native_base_url}/models/unload",
            api_key=self.api_key,
            payload={"instance_id": instance_id},
        )


def get_api_adapter(api_provider: str, url: str, api_key: str = ""):
    if api_provider == "lmstudio":
        return LMStudioApiAdapter(url, api_key)
    if api_provider == "openai":
        return OpenAICompatibleApiAdapter(url, api_key)
    return OllamaApiAdapter(url, api_key)


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")
    api_provider = data.get("api_provider", "ollama")
    api_key = data.get("api_key", "")

    try:
        adapter = get_api_adapter(api_provider, url, api_key)
        models = adapter.list_models()
        return web.json_response(models)
    except Exception as e:
        print(f"Error fetching models from {api_provider} endpoint {url}: {e}")
        return web.json_response([])
    

def unload_model(client, model):
    # if text is empty, or whitespace, return empty string
    if not model or not model.strip():
        return ""

    try:
        print(f"calling Ollama Vision to unload model {model}")
        result = client.generate(
            model=model,
            keep_alive=0,
        )
    except Exception as e:
        print(f"Error calling OllamaVision to unload model {model}: {e}")
    return


 
class OllamaVision:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "images": ("IMAGE",),
                "query": ("STRING", {
                    "multiline": True,
                    "default": "describe the image"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json",''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(self, images, query, debug, url, model, seed, keep_alive, format):
        images_binary = []

        if format == "text":
            format = ''

        for (batch_number, image) in enumerate(images):
            # Convert tensor to numpy array
            i = 255. * image.cpu().numpy()
            
            # Create PIL Image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to BytesIO buffer
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # Get binary data
            img_binary = buffered.getvalue()
            images_binary.append(img_binary)

        client = Client(host=url)

        if debug == "enable":
            print(f"""[Ollama Vision]
request query params:

- query: {query}
- url: {url}
- model: {model}

""")

        response = client.generate(
            model=model,
            prompt=query,
            images=images_binary,
            keep_alive=str(keep_alive) + "m",
            format=format
        )

        if debug == "enable":
            print("[Ollama Vision]\nResponse:\n")
            pprint(response)

        return (response['response'],)

class OllamaVts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "base_positive": ("STRING", {"default": "(photorealistic:1.2) (photo:1.2)", "multiline": True}),
                "base_positive_face": ("STRING", {"default": "a beautiful face with lots of detail and striking eyes", "multiline": True}),
                "base_negative": ("STRING", {"default": "cartoon, cgi, render, illustration, painting, drawing, cartoon, (worst quality, low quality, normal quality:2), out of focus", "multiline": True}),
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "character_face_text": ("STRING", {"default": "", "multiline": True}),
                "character_face_and_body_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_text": ("STRING", {"default": "", "multiline": True}),
                "character_muscle_text": ("STRING", {"default": "", "multiline": True}),
                "character_face_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_tags_text": ("STRING", {"default": "", "multiline": True}),
                "character_ethnicity_tags_text": ("STRING", {"default": "", "multiline": True}),
                "environment_text": ("STRING", {"default": "", "multiline": True}),
                "environment_comma_text": ("STRING", {"default": "", "multiline": True}),
                "environment_images": ("IMAGE",),
                "character_images": ("IMAGE",),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json",''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 100,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                    },
                ),
                "body_tags_multiply": (
                    "FLOAT",
                    {
                        "default": 1.00,
                    },
                ),
                "ethnicity_tags_multiply": (
                    "FLOAT",
                    {
                        "default": 1.00,
                    },
                ),
            },
            "optional": {
                "passthrough": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING"
    )
    OUTPUT_IS_LIST = (
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True
    )

    RETURN_NAMES = (
        "character_face_text",
        "character_face_and_body_text",
        "character_body_text",
        "character_muscle_text",
        "character_face_comma_text",
        "character_comma_text",
        "character_body_tags_text",
        "character_ethnicity_tags_text",
        "environment_text",
        "environment_comma_text",
        "character_neg_body_tags_text",
        "character_neg_ethnicity_tags_text",
        "environment_positive_text",
        "environment_negative_text",
        "character_positive_face_texts",
        "character_negative_face_texts",
        "character_positive_texts",
        "character_negative_texts"
    )

    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    @staticmethod
    def calculate_results(client, model, query: str, split_text: str, images, keep_alive: int, format: str, seed: int, top_p: float, top_k: int, temperature: float, repetition_penalty: float, max_new_tokens: int) -> List[str]:
        # if text is empty, or whitespace, return empty string
        if not query or not query.strip():
            return []

        used_keep_alive = str(keep_alive) + "m"
        if keep_alive == 0:
            used_keep_alive = 0
        
        # split text by newline
        texts = query.split(split_text)
        finalResults = []
        for text in texts:
            try:
                # remove any leading or trailing whitespace
                text = text.strip()
                #remove any leading or trailing newline characters
                text = text.strip("\n")
                # remove any leading or trailing whitespace
                text = text.strip()
                print("calling Ollama Vision")
                result = client.generate(
                    model=model,
                    prompt=text,
                    images=images,
                    keep_alive=used_keep_alive,
                    format=format,
                    options={
                        "seed": seed,
                        "top_k": top_k,
                        "top_p": top_p,
                        "temperature": temperature,
                        "repeat_penalty": repetition_penalty,
                        "num_ctx": max_new_tokens,
                    }
                )
                print("received Ollama Vision Response")
                print(f"""[Ollama Vision]
    - query: {text}
    - result: {result['response']}

    """)
                finalResults.append(result['response'])
            except Exception as e:
                print(f"Error calling OllamaVision: {e}")
        return finalResults
    
    @staticmethod
    def filter_values(input_string: Union[str, List[str]], multiply: float) -> Tuple[str, str]:
        try:
            # Check if the input is a list of strings
            if isinstance(input_string, list):
                # Join the list into a single string with comma and space
                input_string = ', '.join(input_string)

            # Remove any ` or ' characters
            input_string = input_string.replace("`", "").replace("'", "").replace("\r", "").replace("\n", "").strip()
            
            # Split the input string by commas
            value_pairs = input_string.split(', ')
            
            # Initialize lists to store the filtered value pairs
            filtered_values = []
            zero_or_less_values = []
            
            # Iterate through each value pair
            for pair in value_pairs:
                # Extract the key and numerical value from the pair
                key, value = pair.split(':')
                value = float(value.strip(')'))
                
                # Check if the key ends with " arms"
                if key.strip().endswith(" arms"):
                    # Reduce the numerical value
                    value /= 4

                # Check if the key ends with "chinese", "japanese", or "korean"
                if key.strip().endswith("chinese") or key.strip().endswith("japanese") or key.strip().endswith("korean"):
                    # Reduce the numerical value
                    value -= 0.2
                
                # Check if the key ends with "abs"
                if (key.strip().endswith("abs") or "large breasts" in key.lower()) and value > 0.8:
                    # Increase the numerical value
                    value *= 1.25

                # Check if the value is above 1.5
                if value > 1.5:
                    value = 1.5

                if not "breast" in key.lower() and not "abs" in key.lower():
                    value *= multiply

                # Round the value to 2 decimal places
                value = round(value, 2)
                
                # Check if the numerical value is greater than zero
                if value > 0:
                    # If it is, add the pair to the filtered values list
                    filtered_values.append(f"{key}:{value})")
                    if "large breasts" in key.lower() and value > 0.5:
                        filtered_values.append(f"(enormous breasts:{value})")
                        filtered_values.append(f"(big breasts:{value})")
                        filtered_values.append(f"(huge breasts:{value})")
                else:
                    # Otherwise, add the pair to the zero or less values list
                    zero_or_less_values.append(f"{key}:1.2)")
            
            # Join the filtered value pairs back into a comma-separated string
            filtered_string = ', '.join(filtered_values)
            filtered_negative_string = ', '.join(zero_or_less_values)
            return filtered_string, filtered_negative_string
        except Exception as e:
            # Return the original input and an empty string in case of an error
            return input_string, ""

    @staticmethod
    def get_binary_images(images) -> List[bytes]:
        images_binary = []

        for (batch_number, image) in enumerate(images):
            # Convert tensor to numpy array
            i = 255. * image.cpu().numpy()
            
            # Create PIL Image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to BytesIO buffer
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # Get binary data
            img_binary = buffered.getvalue()
            images_binary.append(img_binary)
        return images_binary
    
    @staticmethod
    def to_text(input_data, delimiter: str = "\n") -> str:
        # Replace all occurrences of "\n" (literal newline) in the delimiter with an actual newline character.
        delimiter = delimiter.replace("\\n", "\n")

        # Try to convert input_data to a list or dict if it's a string
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
                print("input_data successfully converted from string to list or dict")
            except json.JSONDecodeError:
                print("input_data is a string but not a valid JSON, proceeding as a single string value")

        if isinstance(input_data, list):
            print("input_data is a list")
            # Convert each element to a string and join them with the delimiter
            merged_text = delimiter.join(str(item) for item in input_data)
        elif isinstance(input_data, dict):
            print("input_data is a dict")
            # Convert each value to a string and join them with the delimiter
            merged_text = delimiter.join(str(value) for value in input_data.values())
        else:
            # Convert the single value to a string
            print(f"input_data is a single value of type {type(input_data).__name__}")
            merged_text = str(input_data)

        # trim the merged text
        merged_text = merged_text.strip()

        return merged_text
    
    @staticmethod
    def get_character_texts(
        client,
        model: str,
        mid_question_alive: int,
        character_image_binary: bytes,
        base_positive: str,
        base_positive_face: str,
        base_negative: str,
        split_text: str,
        character_face_text: str,
        character_face_and_body_text: str,
        character_body_text: str,
        character_muscle_text: str,
        character_face_comma_text: str,
        character_comma_text: str,
        character_body_tags_text: str,
        character_ethnicity_tags_text: str,
        seed: int,
        format: str,                      # Assuming it's either "text", "json", or an empty string
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        body_tags_multiply: float,
        ethnicity_tags_multiply: float,
    ) -> dict[str, str]:
        character_images_binary = [character_image_binary]
        
        character_face_text_results = OllamaVts.calculate_results(client, model, character_face_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        character_face_and_body_text_results = OllamaVts.calculate_results(client, model, character_face_and_body_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        character_body_text_results = OllamaVts.calculate_results(client, model, character_body_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        character_muscle_text_results = OllamaVts.calculate_results(client, model, character_muscle_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        # character_text_results is the character_body_text_results array concatenated to the character_face_text_results array //
        character_full_results = character_face_and_body_text_results + character_face_text_results + character_body_text_results
        character_body_and_muscle_results = character_face_and_body_text_results + character_body_text_results + character_muscle_text_results

        # character_text is the character_text_results array concatenated to a single string with a newline character as the separator and enclosed in ``` characters
        character_full_statement = "```\n" + OllamaVts.to_text(character_full_results) + "\n```"
        character_face_and_body_statement = "```\n" + OllamaVts.to_text(character_face_and_body_text_results) + "\n```"
        character_face_statement = "```\n" + OllamaVts.to_text(character_face_text_results) + "\n```"
        character_body_statement = "```\n" + OllamaVts.to_text(character_body_text_results) + "\n```"
        character_body_and_muscle_results_statement = "```\n" + OllamaVts.to_text(character_body_and_muscle_results) + "\n```"

        used_character_face_comma_text = character_face_comma_text + character_face_statement
        character_face_comma_text_results = OllamaVts.calculate_results(client, model, used_character_face_comma_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        used_character_face_and_body_comma_text = character_comma_text + character_face_and_body_statement
        character_face_and_body_comma_text_results = OllamaVts.calculate_results(client, model, used_character_face_and_body_comma_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        used_character_comma_text = character_comma_text + character_body_statement
        character_comma_text_results = OllamaVts.calculate_results(client, model, used_character_comma_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        used_ethnicity_text = character_ethnicity_tags_text + character_full_statement
        character_ethnicity_tags_text_results, character_ethnicity_tags_text_neg_results = OllamaVts.filter_values(
            OllamaVts.calculate_results(client, model, used_ethnicity_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens),
            ethnicity_tags_multiply
        )

        used_body_tags_text = character_body_tags_text + character_body_and_muscle_results_statement
        character_body_tags_text_results, character_body_tags_text_neg_results = OllamaVts.filter_values(
            OllamaVts.calculate_results(client, model, used_body_tags_text, split_text, character_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens),
            body_tags_multiply
        )

        character_positive_face_text = f"{base_positive}, {base_positive_face}, {OllamaVts.to_text(character_face_and_body_comma_text_results)}, {OllamaVts.to_text(character_face_comma_text_results)}, {OllamaVts.to_text(character_ethnicity_tags_text_results)}"
        character_negative_face_text = f"{base_negative}, {OllamaVts.to_text(character_ethnicity_tags_text_neg_results)}"
        character_positive_text = f"{base_positive}, {OllamaVts.to_text(character_face_and_body_comma_text_results)}, {OllamaVts.to_text(character_comma_text_results)}, {OllamaVts.to_text(character_body_tags_text_results)}, {OllamaVts.to_text(character_ethnicity_tags_text_results)}"
        character_negative_text = f"{base_negative}, {OllamaVts.to_text(character_body_tags_text_neg_results)}, {OllamaVts.to_text(character_ethnicity_tags_text_neg_results)}"

        output_dictionary = {
            "character_face_text": OllamaVts.to_text(character_face_text_results),
            "character_face_and_body_text": OllamaVts.to_text(character_face_and_body_text_results),
            "character_body_text": OllamaVts.to_text(character_body_text_results),
            "character_muscle_text": OllamaVts.to_text(character_muscle_text_results),
            "character_face_comma_text": OllamaVts.to_text(character_face_comma_text_results),
            "character_comma_text": OllamaVts.to_text(character_comma_text_results),
            "character_body_tags_text": OllamaVts.to_text(character_body_tags_text_results),
            "character_ethnicity_tags_text": OllamaVts.to_text(character_ethnicity_tags_text_results),
            "character_neg_body_tags_text": OllamaVts.to_text(character_body_tags_text_neg_results),
            "character_ethnicity_tags_text_neg": OllamaVts.to_text(character_ethnicity_tags_text_neg_results),
            "character_positive_face_text": OllamaVts.to_text(character_positive_face_text),
            "character_negative_face_text": OllamaVts.to_text(character_negative_face_text),
            "character_positive_text": OllamaVts.to_text(character_positive_text),
            "character_negative_text": OllamaVts.to_text(character_negative_text),
        }

        return output_dictionary

    def ollama_vision(
        self,
        base_positive: str,
        base_positive_face: str,
        base_negative: str,
        split_text: str,
        character_face_text: str,
        character_face_and_body_text: str,
        character_body_text: str,
        character_muscle_text: str,
        character_face_comma_text: str,
        character_comma_text: str,
        character_body_tags_text: str,
        character_ethnicity_tags_text: str,
        environment_text: str,
        environment_comma_text: str,
        environment_images,
        character_images,
        debug: str,                       # Assuming it's either "enable" or "disable"
        url: str,
        model: str,                       # Assuming it's a string
        seed: int,
        keep_alive: int,
        format: str,                      # Assuming it's either "text", "json", or an empty string
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        body_tags_multiply: float,
        ethnicity_tags_multiply: float,
        passthrough: bool = False
    ):
        if passthrough:
            return (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                "",
                "",
                [],
                [],
                "",
                "",
                [],
                [],
                [],
                [],)

        if format == "text":
            format = ''

        environment_images_binary = OllamaVts.get_binary_images(environment_images)
        character_images_binary = OllamaVts.get_binary_images(character_images)

        client = Client(host=url)
        if debug == "enable":
            print(f"""[Ollama Vision]
request query params:

- url: {url}
- model: {model}

""")
        mid_question_alive = 4
        character_positive_face_texts = []
        character_negative_face_texts = []
        character_positive_texts = []
        character_negative_texts = []
        character_face_text_results = []
        character_face_and_body_text_results = []
        character_body_text_results = []
        character_muscle_text_results = []
        character_face_comma_text_results = []
        character_comma_text_results = []
        character_body_tags_text_results = []
        character_ethnicity_tags_text_results = []
        character_body_tags_text_neg_results = []
        character_ethnicity_tags_text_neg_results = []

        for character_image_count, character_image_binary in enumerate(character_images_binary):
            results_dictionary = OllamaVts.get_character_texts(
                client,
                model,
                mid_question_alive,
                character_image_binary,
                base_positive,
                base_positive_face,
                base_negative,
                split_text,
                character_face_text,
                character_face_and_body_text,
                character_body_text,
                character_muscle_text,
                character_face_comma_text,
                character_comma_text,
                character_body_tags_text,
                character_ethnicity_tags_text,
                seed,
                format,
                top_p,
                top_k,
                temperature,
                repetition_penalty,
                max_new_tokens,
                body_tags_multiply,
                ethnicity_tags_multiply,
            )
            character_positive_face_texts.append(results_dictionary["character_positive_face_text"])
            character_negative_face_texts.append(results_dictionary["character_negative_face_text"])
            character_positive_texts.append(results_dictionary["character_positive_text"])
            character_negative_texts.append(results_dictionary["character_negative_text"])
            character_face_text_results.append(results_dictionary["character_face_text"])
            character_face_and_body_text_results.append(results_dictionary["character_face_and_body_text"])
            character_body_text_results.append(results_dictionary["character_body_text"])
            character_muscle_text_results.append(results_dictionary["character_muscle_text"])
            character_face_comma_text_results.append(results_dictionary["character_face_comma_text"])
            character_comma_text_results.append(results_dictionary["character_comma_text"])
            character_body_tags_text_results.append(results_dictionary["character_body_tags_text"])
            character_ethnicity_tags_text_results.append(results_dictionary["character_ethnicity_tags_text"])
            character_body_tags_text_neg_results.append(results_dictionary["character_neg_body_tags_text"])
            character_ethnicity_tags_text_neg_results.append(results_dictionary["character_ethnicity_tags_text_neg"])

        environment_text_results = OllamaVts.calculate_results(client, model, environment_text, split_text, environment_images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        environment_text = "```\n" + OllamaVts.to_text(environment_text_results) + "\n```"

        used_environment_comma_text = environment_comma_text + environment_text
        environment_comma_text_results = OllamaVts.calculate_results(client, model, used_environment_comma_text, split_text, None, keep_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        environment_positive_text = f"{base_positive}, {OllamaVts.to_text(environment_comma_text_results)}"
        environment_negative_text = base_negative

        return (
                character_face_text_results,
                character_face_and_body_text_results,
                character_body_text_results,
                character_muscle_text_results,
                character_face_comma_text_results,
                character_comma_text_results,
                character_body_tags_text_results,
                character_ethnicity_tags_text_results,
                environment_text_results,
                environment_comma_text_results,
                character_body_tags_text_neg_results,
                character_ethnicity_tags_text_neg_results,
                environment_positive_text,
                environment_negative_text,
                character_positive_face_texts,
                character_negative_face_texts,
                character_positive_texts,
                character_negative_texts
            )


class OllamaImageQuestionsVts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "api_provider": (["ollama", "openai", "lmstudio"], {"default": "ollama"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully answering questions about images. You are always direct and to the point, answering with confidence and without prefix or postfix text.",
                    "title":"system"
                }),
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "questions": ("STRING", {"default": "", "multiline": True}),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "ordering_input": ("STRING", {"default": "", "multiline": False}),
                "hide_thinking": ("BOOLEAN", {"default": False}),
                "triple_quote_input_text": (["enable", "disable"],),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                # model settings
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "model": ((), {}),
                "format": (["text", "json",''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 100,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                    },
                ),
            },
            "optional":
                    {
                        "images": ("IMAGE",),
                        "passthrough": ("BOOLEAN", {"default": False}),
                        "keepModelLoaded": ("BOOLEAN", {"default": True}),
                    }

        }
    
    INPUT_IS_LIST = True

    RETURN_TYPES = (
        "STRING", # question answers output
        "STRING", # ordering_output
    )
    OUTPUT_IS_LIST = (
        True, # question answers output is a list of string, 1 item per image
        False, # ordering_output is a single string
    )

    RETURN_NAMES = (
        "answers", # question answers output
        "ordering_output", # ordering_output
    )

    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    @staticmethod
    def calculate_results(adapter, model, system: str, query: str, split_text: str, images, input_text: str, keep_alive: int, format: str, seed: int, top_p: float, min_p: float, top_k: int, temperature: float, repetition_penalty: float, max_new_tokens: int, debug: str, hide_thinking: bool) -> List[str]:
        # if text is empty, or whitespace, return empty string
        if not query or not query.strip():
            return ""

        # split text by newline
        texts = query.split(split_text)
        finalResults = []
        for text in texts:
            try:
                # remove any leading or trailing whitespace
                text = text.strip()
                #remove any leading or trailing newline characters
                text = text.strip("\n")
                # remove any leading or trailing whitespace
                text = text.strip()
                if input_text:
                    text = f"{text}\n{input_text}"
                print("calling image questions provider")
                result = adapter.generate_image_question(
                    model=model,
                    system=system,
                    prompt=text,
                    images=images,
                    keep_alive=keep_alive,
                    format=format,
                    seed=seed,
                    top_k=top_k,
                    min_p=min_p,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    debug=debug,
                )
                if hide_thinking:
                    result = strip_thinking_blocks(result)
                print("received image questions provider response")
                print(f"""[Image Questions]
    - query: {text}
    - result: {result}

    """)
                finalResults.append(result)
            except Exception as e:
                print(f"Error calling image questions provider: {e}")
        return finalResults
    
    @staticmethod
    def filter_values(input_string: Union[str, List[str]], multiply: float) -> Tuple[str, str]:
        try:
            # Check if the input is a list of strings
            if isinstance(input_string, list):
                # Join the list into a single string with comma and space
                input_string = ', '.join(input_string)

            # Remove any ` or ' characters
            input_string = input_string.replace("`", "").replace("'", "").replace("\r", "").replace("\n", "").strip()
            
            # Split the input string by commas
            value_pairs = input_string.split(', ')
            
            # Initialize lists to store the filtered value pairs
            filtered_values = []
            zero_or_less_values = []
            
            # Iterate through each value pair
            for pair in value_pairs:
                # Extract the key and numerical value from the pair
                key, value = pair.split(':')
                value = float(value.strip(')'))
                
                # Check if the key ends with " arms"
                if key.strip().endswith(" arms"):
                    # Reduce the numerical value
                    value /= 4

                # Check if the key ends with "chinese", "japanese", or "korean"
                if key.strip().endswith("chinese") or key.strip().endswith("japanese") or key.strip().endswith("korean"):
                    # Reduce the numerical value
                    value -= 0.2
                
                # Check if the key ends with "abs"
                if (key.strip().endswith("abs") or "large breasts" in key.lower()) and value > 0.8:
                    # Increase the numerical value
                    value *= 1.25

                # Check if the value is above 1.5
                if value > 1.5:
                    value = 1.5

                if not "breast" in key.lower() and not "abs" in key.lower():
                    value *= multiply

                # Round the value to 2 decimal places
                value = round(value, 2)
                
                # Check if the numerical value is greater than zero
                if value > 0:
                    # If it is, add the pair to the filtered values list
                    filtered_values.append(f"{key}:{value})")
                    if "large breasts" in key.lower() and value > 0.5:
                        filtered_values.append(f"(enormous breasts:{value})")
                        filtered_values.append(f"(big breasts:{value})")
                        filtered_values.append(f"(huge breasts:{value})")
                else:
                    # Otherwise, add the pair to the zero or less values list
                    zero_or_less_values.append(f"{key}:1.2)")
            
            # Join the filtered value pairs back into a comma-separated string
            filtered_string = ', '.join(filtered_values)
            filtered_negative_string = ', '.join(zero_or_less_values)
            return filtered_string, filtered_negative_string
        except Exception as e:
            # Return the original input and an empty string in case of an error
            return input_string, ""

    @staticmethod
    def get_binary_images(images) -> List[bytes]:
        images_binary = []
        if images == None:
            return images_binary

        for (batch_number, image) in enumerate(images):
            # Convert tensor to numpy array
            i = 255. * image.cpu().numpy()
            
            # Create PIL Image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to BytesIO buffer
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # Get binary data
            img_binary = buffered.getvalue()
            images_binary.append(img_binary)
        return images_binary
    
    @staticmethod
    def to_text(input_data, delimiter: str = "\n") -> str:
        # Replace all occurrences of "\n" (literal newline) in the delimiter with an actual newline character.
        delimiter = delimiter.replace("\\n", "\n")

        # Try to convert input_data to a list or dict if it's a string
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
                print("input_data successfully converted from string to list or dict")
            except json.JSONDecodeError:
                print("input_data is a string but not a valid JSON, proceeding as a single string value")

        if isinstance(input_data, list):
            print("input_data is a list")
            # Convert each element to a string and join them with the delimiter
            merged_text = delimiter.join(str(item) for item in input_data)
        elif isinstance(input_data, dict):
            print("input_data is a dict")
            # Convert each value to a string and join them with the delimiter
            merged_text = delimiter.join(str(value) for value in input_data.values())
        else:
            # Convert the single value to a string
            print(f"input_data is a single value of type {type(input_data).__name__}")
            merged_text = str(input_data)

        # trim the merged text
        merged_text = merged_text.strip()

        return merged_text
    
    def ollama_vision(
        self,
        api_provider: str,
        api_key: str,
        system: str,
        split_text: str,
        questions: str,
        input_text: str,
        ordering_input: str,
        hide_thinking: bool,
        triple_quote_input_text: str,
        debug: str,                       # Assuming it's either "enable" or "disable"
        url: str,
        keep_alive: int,
        model: str,                       # Assuming it's a string
        format: str,                      # Assuming it's either "text", "json", or an empty string
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        images = [],
        passthrough: bool = False,
        keepModelLoaded: bool = True,
    ):
        
        print(f"len(images): {len(images)}")
        # as input is list, we need to manually set each item that should not be a list to the first item
        if len(images) > 0:
            images = images[0]
        else:
            images = None
        api_provider = api_provider[0]
        api_key = api_key[0]
        system = system[0]
        split_text = split_text[0]
        questions = questions[0]
        # input_text = input_text[0] - don't do input_text as it is valid for it to be a list
        ordering_input = ordering_input[0]
        hide_thinking = hide_thinking[0]
        triple_quote_input_text = triple_quote_input_text[0]
        debug = debug[0]
        url = url[0]
        keep_alive = keep_alive[0]
        model = model[0]
        format = format[0]
        seed = seed[0]
        top_p = top_p[0]
        min_p = min_p[0]
        top_k = top_k[0]
        temperature = temperature[0]
        repetition_penalty = repetition_penalty[0]
        max_new_tokens = max_new_tokens[0]
        passthrough = passthrough[0]
        keepModelLoaded = keepModelLoaded[0]

        if passthrough:
            adapter = get_api_adapter(api_provider, url, api_key)
            if adapter.should_explicitly_unload(keep_alive, keepModelLoaded):
                # Some providers can unload even when no request is made in this pass.
                try:
                    print(f"Attempting to unload model for provider {api_provider}")
                    adapter.maybe_unload_model(model, debug)
                except Exception as e:
                    print(f"Error unloading model: {e}")
            else:
                print(f"keepModelLoaded enabled, not unloading model")
            print(f"Passthrough enabled, returning empty outputs")
            return ([""], ordering_input)

        print(f"system: {system}")

        if format == "text":
            format = ''

        images_binary = OllamaImageQuestionsVts.get_binary_images(images)

        adapter = get_api_adapter(api_provider, url, api_key)
        should_explicitly_unload = adapter.should_explicitly_unload(keep_alive, keepModelLoaded)
        if debug == "enable":
            print(f"""[Image Questions]
request query params:

- api_provider: {api_provider}
- url: {url}
- model: {model}

""")

        mid_question_alive = 5
        answers = []
        number_of_images = len(images_binary)
        number_of_input_texts = len(input_text)
        max_length = max(number_of_images, number_of_input_texts)

        print(f"number_of_images: {number_of_images}")
        print(f"number_of_input_texts: {number_of_input_texts}")
        print(f"max_length: {max_length}")
        print(f"input_text: {input_text}")

        for i in range(max_length):
            usedInputTextIndex = min(i, number_of_input_texts - 1)
            usedInputImageIndex = min(i, number_of_images - 1)
            print(f"i: {i}")
            print(f"usedInputTextIndex: {usedInputTextIndex}")
            print(f"usedInputImageIndex: {usedInputImageIndex}")
            used_input_text = input_text[usedInputTextIndex] if usedInputTextIndex >= 0 else None
            image_binary = images_binary[usedInputImageIndex] if usedInputImageIndex >= 0 else None

            if used_input_text:
                print(f"used_input_text: {used_input_text}")
                used_input_text = used_input_text.strip()
                if triple_quote_input_text == "enable":
                    used_input_text = f"```{used_input_text}```"

            if image_binary:
                image_binary = [image_binary]

            used_keep_alive = mid_question_alive
            if i == max_length - 1:
                used_keep_alive = keep_alive
            answer = OllamaImageQuestionsVts.calculate_results(adapter, model, system, questions, split_text, image_binary, used_input_text, used_keep_alive, format, seed, top_p, min_p, top_k, temperature, repetition_penalty, max_new_tokens, debug, hide_thinking)
            answers.append(OllamaImageQuestionsVts.to_text(answer))

        ordering_output = ordering_input
        if len(answers) == 0:
            answers.append("")

        if should_explicitly_unload:
            try:
                print(f"Attempting to unload model for provider {api_provider}")
                adapter.maybe_unload_model(model, debug)
            except Exception as e:
                print(f"Error unloading model: {e}")
        else:
            print(f"keepModelLoaded enabled, not unloading model")
        return (answers, ordering_output)

class OllamaSettingsVts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully answering questions about images. You are always direct and to the point, answering with confidence and without prefix or postfix text.",
                    "title":"system"
                }),
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "ordering_input": ("STRING", {"default": "", "multiline": False}),
                "triple_quote_input_text": (["enable", "disable"],),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                # model settings
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "model": ((), {}),
                "format": (["text", "json",''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 100,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 2048,
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING", 
        "STRING", 
        "STRING", 
        "STRING", 
        "STRING", 
        "INT", 
        "STRING", 
        "STRING", 
        "INT",
        "FLOAT",
        "FLOAT",
        "INT",
        "FLOAT",
        "FLOAT",
        "INT",
    )

    RETURN_NAMES = (
        "system",
        "split_text",
        "ordering_input",
        "triple_quote_input_text",
        "debug",
        "url",
        "keep_alive",
        "model",
        "format",
        "seed",
        "top_p",
        "min_p",
        "top_k",
        "temperature",
        "repetition_penalty",
        "max_new_tokens",
    )

    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(
        self,
        system: str,
        split_text: str,
        ordering_input: str,
        triple_quote_input_text: str,
        debug: str,                       # Assuming it's either "enable" or "disable"
        url: str,
        keep_alive: int,
        model: str,                       # Assuming it's a string
        format: str,                      # Assuming it's either "text", "json", or an empty string
        seed: int,
        top_p: float,
        min_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
    ):

        return (system,
                split_text,
                ordering_input,
                triple_quote_input_text,
                debug,
                url,
                keep_alive,
                model,
                format,
                seed,
                top_p,
                min_p,
                top_k,
                temperature,
                repetition_penalty,
                max_new_tokens,
                )



class OllamaGenerate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json",''],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model, keep_alive, format):

        client = Client(host=url)

        if format == "text":
            format = ''

        if debug == "enable":
            print(f"""[Ollama Generate]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}

            """)

        response = client.generate(model=model, prompt=prompt, keep_alive=str(keep_alive) + "m", format=format)

        if debug == "enable":
                print("[Ollama Generate]\nResponse:\n")
                pprint(response)

        return (response['response'],)

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

class OllamaGenerateAdvance:
    saved_context = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                    "title":"system"
                }),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "format": (["text", "json",''],),
            },"optional": {
                "context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "context",)
    FUNCTION = "ollama_generate_advance"
    CATEGORY = "Ollama"

    def ollama_generate_advance(self, prompt, debug, url, model, system, seed, top_k, top_p,temperature, num_predict, tfs_z, keep_alive, keep_context, format, context=None):

        client = Client(host=url)

        if format == "text":
            format = ''

        # num_keep: int
        # seed: int
        # num_predict: int
        # top_k: int
        # top_p: float
        # tfs_z: float
        # typical_p: float
        # repeat_last_n: int
        # temperature: float
        # repeat_penalty: float
        # presence_penalty: float
        # frequency_penalty: float
        # mirostat: int
        # mirostat_tau: float
        # mirostat_eta: float
        # penalize_newline: bool
        # stop: Sequence[str]

        options = {
            "seed": seed,
            "top_k":top_k,
            "top_p":top_p,
            "temperature":temperature,
            "num_predict":num_predict,
            "tfs_z":tfs_z,
        }

        if context != None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context == None:
            context = self.saved_context

        if debug:
            print(f"""[Ollama Generate Advance]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- options: {options}
""")

        response = client.generate(model=model, system=system, prompt=prompt, context=context, options=options, keep_alive=str(keep_alive) + "m", format=format)
        if debug:
            print("[Ollama Generate Advance]\nResponse:\n")
            pprint(response)

        if keep_context:
            self.saved_context = response["context"]

        return (response['response'], response['context'],)

class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path + ".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max":2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),

                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),

                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),

                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the size of the context window used to generate the next token."}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = num_ctx)"}),

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),

                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),

                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return."}),

                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),

                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),

                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out."}),

                "debug": ("BOOLEAN", {"default": False, "tooltip": "For debugging purposes of the custom nodes, no effect on ollama api."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"
    DESCRIPTION = "Various settings for advanced configuration of Ollama inference. See Ollama documentation for more details."

    def ollama_options(self, **kargs):

        if kargs['debug']:
            print("--- ollama options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs,)

class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }

        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'"
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation. This is useful for multi-turn conversations or tasks that require context."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."}),

            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False, "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set."},),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False, "tooltip": "Connect an Ollama Options node for advanced inference configuration."},),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."},),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."},),
                "meta": ("OLLAMA_META", {"forceInput": False, "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along."},),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def get_request_options(self, options):
        response = None

        if options is None:
            return response

        enablers = ['enable_mirostat', 'enable_mirostat_eta',
                    'enable_mirostat_tau', 'enable_mirostat_eta',
                    'enable_num_ctx', 'enable_repeat_last_n', 'enable_repeat_penalty',
                    'enable_temperature', 'enable_seed', 'enable_stop', 'enable_tfs_z', 'enable_num_predict',
                    'enable_top_k', 'enable_top_p', 'enable_min_p']

        for enabler in enablers:
            if options[enabler]:
                if response is None:
                    response = {}
                key = enabler.replace("enable_", "")
                response[key] = options[key]

        return response

    def ollama_generate_v2(self, system, prompt, think, keep_context, format, context = None, options=None, connectivity=None, images=None, meta=None):

        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None: # bypass the current meta connectivity
                meta["connectivity"] = connectivity
            if options is not None: # bypass the current meta options
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        url = meta['connectivity']['url']
        model = meta['connectivity']['model']
        client = Client(host=url)

        debug_print = True if meta['options'] is not None and meta['options']['debug'] else False

        if format == "text":
            format = ''

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit =  'm' if meta['connectivity']['keep_alive_unit'] == "minutes" else 'h'
        request_keep_alive = str(meta['connectivity']['keep_alive']) + keep_alive_unit

        request_options = self.get_request_options(options)

        images_b64 = None
        if images is not None:
            images_b64 = []
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                images_b64.append(str(img_bytes, 'utf-8'))

        if debug_print:
            print(f"""
--- ollama generate v2 request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
context: {context}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
""")

        response = client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images_b64,
            context=context,
            think=think,
            options=request_options,
            keep_alive= request_keep_alive,
            format=format,
        )

        if debug_print:
            print("\n--- ollama generate v2 response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response['response']
        ollama_response_thinking = response['thinking'] if think else None

        if keep_context:
            self.saved_context = response["context"]
            if debug_print:
                print("saving context to node memory.")

        return ollama_response_text, ollama_response_thinking, response['context'], meta,


class OllamaChat:
    """
    Text generation with Ollama Chat.
    Returns: (result: str, thinking: str|None, meta: dict, history: str)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json"],
                    {
                        "tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response.",
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Optionally set an existing model history, useful for multi-turn conversations, follow-up questions.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        # Update with provided values (override)
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
        else:
            meta["options"] = None

        # Final validation
        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        client = Client(host=url)

        debug_print = (
            True if meta["options"] is not None and meta["options"]["debug"] else False
        )

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None

        if format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        # 4. use the shared helper instead of self.get_request_options
        request_options = _filter_enabled_options(options)

        images_b64: list[str] | None = None
        if images is not None:
            images_b64 = []
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_bytes)

        if debug_print:
            print(
                f"""
--- ollama chat request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
"""
            )

        # Determinate which session to use
        session_key = history if history is not None else unique_id

        # If reset_session is True, reset the session
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")

        # If the session doesn't exist, create it
        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()

        session = CHAT_SESSIONS[session_key]

        # Update history for return
        history = session_key

        # If there is a system prompt, replace it or add it to the beginning
        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        # Construct the user message for history
        user_message_for_history: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }

        # Add the user message to the history (without images)
        session.messages.append(user_message_for_history)

        if debug_print:
            print("\n--- ollama chat session:")
            for message in session.messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
                if "images" in message:
                    for image in message["images"]:
                        pprint(f"Image: {image[:50]}...")
            print("---------------------------------------------------------")

        # Construct the messages for the API call (with images)
        messages_for_api = copy.deepcopy(session.messages)

        # If there are images, modify the last user message for the API call
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        response = client.chat(
            model=model,
            messages=messages_for_api,
            options=request_options,
            keep_alive=request_keep_alive,
            format=ollama_format,
        )

        if debug_print:
            print("\n--- ollama chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response.message.content
        ollama_response_thinking = response.message.thinking if think else None

        # Add the assistant message to the history
        session.messages.append(
            {
                "role": "assistant",
                "content": ollama_response_text,
            }
        )

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            history,
        )


NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaVts": OllamaVts,
    "OllamaImageQuestionsVts": OllamaImageQuestionsVts,
    "OllamaSettingsVts": OllamaSettingsVts,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
    "OllamaChat": OllamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaVts": "Ollama Vts",
    "OllamaImageQuestionsVts": "Ollama Image Questions Vts",
    "OllamaSettingsVts": "Ollama Settings Vts",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
    "OllamaOptionsV2": "Ollama Options",
    "OllamaConnectivityV2": "Ollama Connectivity",
    "OllamaGenerateV2": "Ollama Generate",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
    "OllamaChat": "Ollama Chat",
}
