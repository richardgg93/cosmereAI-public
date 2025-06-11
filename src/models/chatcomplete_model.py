import requests

from typing import List, Dict
import json


class AzureAIChatComplete:
    """Loads or create chat complete models"""

    def __init__(
        self,
        endpoint: str = "",
        token: str = "",
        model_name: str = "",
    ) -> None:
        """ """
        self.endpoint = endpoint
        self.token = token
        self.model_name = model_name

    def predict(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """Transform a list of strings into embeddings using model.

        Args:
            input_data (List[str]): list of strings.

        Returns:
            np.array: Array with the embeddings.
        """
        data = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 512),
        }

        chatcomplete_endpoint = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.token),
        }

        response = requests.post(chatcomplete_endpoint, headers=headers, json=data)

        return json.loads(response.text)
