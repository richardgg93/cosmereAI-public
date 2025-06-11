import requests

import io
import base64
from PIL import Image


class StabilityAIImageGen:
    """Loads or create models for image generation"""

    def __init__(
        self,
        endpoint: str = "",
        token: str = "",
        model_name: str = "sd3-turbo",
    ) -> None:
        """ """
        self.endpoint = endpoint
        self.token = token
        self.model_name = model_name

    def predict(self, prompt: str, **kwargs) -> Image:
        """Generate an image from a prompt description using AI model.

        Args:
            prompt (str): Image description.

        Returns:
            Image: Array with the image.
        """
        data = {
            "prompt": prompt,
            "model": self.model_name,
            "style_preset": kwargs.get("style_preset", None),
            "negative_prompt": "Be careful with the errors in the generated images. Hands and faces should look right. And text should be properly written.",
        }

        imagegen_endpoint = f"{self.endpoint}"
        headers = {
            "Authorization": ("Bearer " + self.token),
            "accept": "application/json",
        }
        files = {"none": ""}

        response = requests.post(imagegen_endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            img = Image.open(io.BytesIO(base64.b64decode(response.json()["image"])))
            return img
        else:
            raise Exception(str(response.json()))
