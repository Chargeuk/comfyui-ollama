import random
from typing import Tuple, Union, List
import json

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
            return ""

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

        # character_text_results is the character_body_text_results array concatenated to the character_face_text_results array
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
    ):
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
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully answering questions about images. You are always direct and to the point, answering with confidence and without prefix or postfix text.",
                    "title":"system"
                }),
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "questions": ("STRING", {"default": "", "multiline": True}),
                "input_text": ("STRING", {"default": "", "multiline": True}),
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
            },
            "optional":
                    {"images": ("IMAGE",),},
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
    def calculate_results(client, model, system: str, query: str, split_text: str, images, input_text: str, keep_alive: int, format: str, seed: int, top_p: float, min_p: float, top_k: int, temperature: float, repetition_penalty: float, max_new_tokens: int) -> List[str]:
        # if text is empty, or whitespace, return empty string
        if not query or not query.strip():
            return ""

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
                if input_text:
                    text = f"{text}\n{input_text}"
                print("calling Ollama Vision")
                result = client.generate(
                    model=model,
                    system=system,
                    prompt=text,
                    images=images,
                    keep_alive=used_keep_alive,
                    format=format,
                    options={
                        "seed": seed,
                        "top_k": top_k,
                        "min_p": min_p,
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
        system: str,
        split_text: str,
        questions: str,
        input_text: str,
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
        images = [],
    ):
        print(f"len(images): {len(images)}")
        # as input is list, we need to manually set each item that should not be a list to the first item
        if len(images) > 0:
            images = images[0]
        else:
            images = None
        system = system[0]
        split_text = split_text[0]
        questions = questions[0]
        # input_text = input_text[0] - don't do input_text as it is valid for it to be a list
        ordering_input = ordering_input[0]
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

        print(f"system: {system}")

        if format == "text":
            format = ''

        images_binary = OllamaImageQuestionsVts.get_binary_images(images)

        client = Client(host=url)
        if debug == "enable":
            print(f"""[Ollama Vision]
request query params:

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
                used_input_text = f"```{used_input_text}```"

            if image_binary:
                image_binary = [image_binary]

            used_keep_alive = mid_question_alive
            if i == max_length - 1:
                used_keep_alive = keep_alive
            answer = OllamaImageQuestionsVts.calculate_results(client, model, system, questions, split_text, image_binary, used_input_text, used_keep_alive, format, seed, top_p, min_p, top_k, temperature, repetition_penalty, max_new_tokens)
            answers.append(OllamaImageQuestionsVts.to_text(answer))

        ordering_output = ordering_input
        return (answers, ordering_output)


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
    "OllamaImageQuestionsVts": OllamaImageQuestionsVts,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaVts": "Ollama Vts",
    "OllamaImageQuestionsVts": "Ollama Image Questions Vts",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
}
