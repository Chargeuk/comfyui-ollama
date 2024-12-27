import random

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


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")
    client = Client(host=url)

    models = client.list().get('models', [])

    try:
        models = [model['model'] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model['name'] for model in models]
        return web.json_response(models)

 
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
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "character_face_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_text": ("STRING", {"default": "", "multiline": True}),
                "character_muscle_text": ("STRING", {"default": "", "multiline": True}),
                "character_face_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_tags_text": ("STRING", {"default": "", "multiline": True}),
                "character_ethnicity_tags_text": ("STRING", {"default": "", "multiline": True}),
                "environment_text": ("STRING", {"default": "", "multiline": True}),
                "environment_comma_text": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE",),
                # "query": ("STRING", {
                #     "multiline": True,
                #     "default": "describe the image"
                # }),
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
                    },
                ),
            },
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
        "STRING"
    )

    RETURN_NAMES = (
        "character_face_text",
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
    )
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def calculate_results(self, client, model, query, split_text, images, keep_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens):
        # if text is empty, or whitespace, return empty string
        if not query or not query.strip():
            return ""
        
        # split text by newline
        texts = query.split(split_text)
        finalResults = []
        for text in texts:
            # remove any leading or trailing whitespace
            text = text.strip()
            #remove any leading or trailing newline characters
            text = text.strip("\n")
            # remove any leading or trailing whitespace
            text = text.strip()

            result = client.generate(
                model=model,
                prompt=text,
                images=images,
                keep_alive=str(keep_alive) + "m",
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
            print(f"""[Ollama Vision]
- query: {text}
- result: {result['response']}

""")
            finalResults.append(result['response'])
        return finalResults

    def filter_values(self, input_string):
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
                if key.strip().endswith("abs"):
                    # Increase the numerical value
                    value *= 1.25

                # Round the value to 2 decimal places
                value = round(value, 2)

                # Check if the value is above 1.5
                if value > 1.5:
                    value = 1.5
                
                # Check if the numerical value is greater than zero
                if value > 0:
                    # If it is, add the pair to the filtered values list
                    filtered_values.append(f"{key}:{value})")
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


    def ollama_vision(
        self,
        split_text,
        character_face_text,
        character_body_text,
        character_muscle_text,
        character_face_comma_text,
        character_comma_text,
        character_body_tags_text,
        character_ethnicity_tags_text,
        environment_text,
        environment_comma_text,
        images,
        debug,
        url,
        model,
        seed,
        keep_alive,
        format,
        top_p,
        top_k,
        temperature,
        repetition_penalty,
        max_new_tokens
    ):
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

- url: {url}
- model: {model}

""")
        mid_question_alive = 5
        # response = client.generate(model=model, prompt=query, images=images_binary, keep_alive=str(keep_alive) + "m", format=format)
        character_face_text_results = self.calculate_results(client, model, character_face_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        character_body_text_results = self.calculate_results(client, model, character_body_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        character_muscle_text_results = self.calculate_results(client, model, character_muscle_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        # character_text_results is the character_body_text_results array concatenated to the character_face_text_results array
        character_text_results = character_face_text_results + character_body_text_results
        character_body_muscle_results = character_body_text_results + character_muscle_text_results
        character_text_muscle_results = character_face_text_results + character_body_text_results + character_muscle_text_results
        # character_text is the character_text_results array concatenated to a single string with a newline character as the separator and enclosed in ``` characters
        character_text = "```\n" + "\n".join(character_text_results) + "\n```"
        character_full_text = "```\n" + "\n".join(character_text_muscle_results) + "\n```"
        character_face_text = "```\n" + "\n".join(character_face_text_results) + "\n```"
        character_body_text = "```\n" + "\n".join(character_body_text_results) + "\n```"
        character_body_muscle_text = "```\n" + "\n".join(character_body_muscle_results) + "\n```"

        used_character_face_comma_text = character_comma_text + character_face_text
        character_face_comma_text_results = self.calculate_results(client, model, used_character_face_comma_text, split_text, None, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        used_character_comma_text = character_face_comma_text + character_text
        character_comma_text_results = self.calculate_results(client, model, used_character_comma_text, split_text, None, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        used_ethnicity_text = character_ethnicity_tags_text + character_text
        character_ethnicity_tags_text_results, character_ethnicity_tags_text_neg_results = self.filter_values(self.calculate_results(client, model, used_ethnicity_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens))

        used_body_tags_text = character_body_tags_text + character_body_muscle_text
        character_body_tags_text_results, character_body_tags_text_neg_results = self.filter_values(self.calculate_results(client, model, used_body_tags_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens))

        environment_text_results = self.calculate_results(client, model, environment_text, split_text, images_binary, mid_question_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
        environment_text = "```\n" + "\n".join(environment_text_results) + "\n```"

        used_environment_comma_text = environment_comma_text + environment_text
        environment_comma_text_results = self.calculate_results(client, model, used_environment_comma_text, split_text, None, keep_alive, format, seed, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

        return (
                character_face_text_results,
                character_body_text_results,
                character_muscle_text_results,
                character_face_comma_text_results,
                character_comma_text_results,
                character_body_tags_text_results,
                character_ethnicity_tags_text_results,
                environment_text_results,
                environment_comma_text_results,
                character_body_tags_text_neg_results,
                character_ethnicity_tags_text_neg_results
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
                    {"context": ("STRING", {"forceInput": True}, ),
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

        image.save(path+".png", pnginfo=metadata)

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

NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaVts": OllamaVts,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaVts": "Ollama Vts",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
}
