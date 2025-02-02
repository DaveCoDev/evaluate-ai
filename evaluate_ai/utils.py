from enum import Enum
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger
import pyarrow.parquet as pq
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from evaluate_ai.constants import AZURE_OPENAI_CLIENT, OLLAMA_CLIENT, OPENAI_CLIENT


class Provider(Enum):
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    OPENAI = "openai_api"


def get_llm_client(provider_name: str) -> Any:
    if provider_name == Provider.OLLAMA.value:
        llm_client = OLLAMA_CLIENT
    elif provider_name == Provider.OPENAI.value:
        llm_client = OPENAI_CLIENT
    elif provider_name == Provider.AZURE_OPENAI.value:
        llm_client = AZURE_OPENAI_CLIENT
    else:
        raise ValueError(f"Provider {provider_name} is not supported.")

    return llm_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def download_file(url: str, file_name: str) -> Path:
    """Download a file from a URL with retries and return the path to the cached file.

    Args:
        url: The URL to download the file from.
        file_name: The name of the file to cache the downloaded file as.

    Returns:
        The path to the cached file.
    """
    temp_file = Path(__file__).parents[1] / "data" / "temp" / file_name
    if not temp_file.exists():
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading data from {url} to {temp_file}.")

        response = requests.get(url)
        response.raise_for_status()
        response = response.content
        with Path.open(temp_file, "wb") as file:
            file.write(response)
    else:
        logger.info(f"File {temp_file} already exists, using cached file.")
    return temp_file


def download_parquet(url: str, file_name: str) -> list[dict]:
    """Download a parquet file from a URL and return it as a list of dictionaries.

    Args:
        url: The URL to download the parquet file from.
        file_name: The name of the file to cache the parquet file as.

    Returns:
        The parquet file contents as a list of dictionaries
    """

    temp_file = download_file(url, file_name)
    with Path.open(temp_file, "rb") as file:
        table = pq.read_table(file)
        return table.to_pylist()


def download_jsonl(url: str, file_name: str) -> list[dict]:
    """Download a jsonl file from a URL and return it as a list of dictionaries.

    Args:
        url: The URL to download the jsonl file from.
        file_name: The name of the file to cache the jsonl file as.

    Returns:
        The jsonl file contents as a list of dictionaries
    """
    temp_file = download_file(url, file_name)
    with Path.open(temp_file, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def strip_thinking(response: str) -> str:
    """Remove text between <think> and </think> tags, including the tags themselves.
    Usually there is unnecessary newlines after the closing </think> tag so we remove it if the think tag is present.

    Args:
        response: The input string that may contain thinking tags.

    Returns:
        The input string with all thinking tags and their contents removed.
    """
    result = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return result.lstrip() if result != response else response
