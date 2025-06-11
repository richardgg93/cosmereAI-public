import requests

from typing import List
import numpy as np
import json

from tenacity import retry, wait_fixed, stop_after_attempt


class AzureAIEmbedding:
    """Loads or create embeddings model,
    that can be used to map sentences / text to embeddings"""

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

    @retry(wait=wait_fixed(60), stop=stop_after_attempt(3))
    def raw_predict(self, input_data: List[str], **kwargs) -> np.array:
        """Transform a list of strings into embeddings using model.

        Args:
            input_data (List[str]): list of strings.

        Returns:
            np.array: Array with the embeddings.
        """
        data = {"input": input_data}

        embeddings_endpoint = f"{self.endpoint}/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.token),
        }

        response = requests.post(embeddings_endpoint, headers=headers, json=data)

        embeddings = np.array([e["embedding"] for e in json.loads(response.text)["data"]])

        return embeddings

    def predict(self, input_data: List[str], **kwargs) -> np.array:
        """Transform a list of strings into embeddings using model.

        Args:
            input_data (List[str]): list of strings.

        Returns:
            np.array: Array with the embeddings.
        """
        embeddings_raw_list = []
        for i in range(0, len(input_data), 96):
            embeddings_raw_list += [self.raw_predict(input_data[i : i + 96])]

        embeddings = np.vstack(embeddings_raw_list)

        return embeddings
