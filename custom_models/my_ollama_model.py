import requests
import json
from typing import List, Any

# Correct imports for the current version of the library
from evals.api import CompletionFn, CompletionResult

# Default Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# The new library version requires a specific result object.
# This simple class formats the response from Ollama correctly.
class OllamaCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any):
        self._raw_data = raw_data

    def get_completions(self) -> List[str]:
        # The framework expects a list of strings as the final output.
        # In our simple case, this list will have just one item.
        response_text = self._raw_data.get("response", "").strip()
        return [response_text]

# The model class MUST inherit from CompletionFn
class OllamaModel(CompletionFn):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    # The main method for a CompletionFn is __call__
    def __call__(self, prompt: str, **kwargs) -> CompletionResult:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            # We return an instance of our custom result class
            return OllamaCompletionResult(response_data)

        except requests.exceptions.RequestException as e:
            error_message = f"ERROR calling Ollama API: {e}"
            print(error_message)
            return OllamaCompletionResult({"response": error_message})