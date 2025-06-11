import sys
sys.path.append("..")
sys.path.append(".")

import json
from typing import List, Dict
from PIL import Image

from azure.search.documents import SearchClient  


from models.embedding_model import AzureAIEmbedding
from models.chatcomplete_model import AzureAIChatComplete
from models.imagegen_model import StabilityAIImageGen

from search_utils import search_knowledgebase_single


class ChatBot:
    """
    Represents a chatbot powered by LLM services.
    """

    ASSISTANT = """You are an assistant with access to the following functions.

    {tools}

    When a function is needed you should generate only a valid JSON with the format:
    {{ 
        "function_name": "name of the function"
        "parameters": [
                    "parameter1": "value of the parameter",
                    "parameter2": "value of the parameter",
                    ...
                    ]
    }}

    Follow the next steps:
    - Decide whether a function is needed or not.
    - If not needed answer the user directly.
    - If a function is needed provide the JSON only.
    """

    ASSISTANT_FINISHER = """You are an assistant. You have last messages of user and tools calling. Provide best answer to the user."""

    FILTERER = """Necesitas responder a una pregunta. Para ello vas a recibir información adicional.
    Tu tarea es filtrar dicha información, quedándote solo con aquella que pueda ser relevante
    para responder la pregunta, y descartando lo demás.
    Responde solamente "SI" o "NO" en función de si la información filtrada es útil o no.

    Pregunta: {pregunta}

    Información a filtrar:
    {info_sin_filtrar}

    ¿Es útil o relevante?:"""

    FUNC_SEARCH_DATA = {
        "name": "ask_data",
        "description": "Auxiliary function to search for specific excerpts from Brandon Sanderson's books. It has all the excerpts separately accessible for searching (in Spanish).",
        "parameters": {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "List of questions. Each element of the list should be a question in the appropriate format."
                    + "For example: ['Who are Kaladin's companions on bridge 4?']",
                }
            },
            "required": ["search_queries"],
        },
    }

    FUNC_CREATE_IMAGE = {
            "name": "create_image",
            "description": "Auxiliary function to generate images from a description using an artificial intelligence model like DALL·E. This model only works in English.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_description": {
                        "type": "string",
                        "description": "Description of the image that should be generated with the AI model. As detailed as possible. This parameter should be always in English.",
                    },
                    "style_preset": {
                        "type": "string",
                        "description": "A style preset to guide the image model towards a particular style. It must be one of: '3d-model', 'analog-film', 'anime', 'cinematic', 'comic-book', 'digital-art', 'enhance', 'fantasy-art', 'isometric', 'line-art', 'low-poly', 'modeling-compound', 'neon-punk', 'origami', 'photographic', 'pixel-art', 'tile-texture'.",
                    }
                },
                "required": ["prompt_description"],
            },
        }

    ASSISTANT_TOOLS = [FUNC_SEARCH_DATA, FUNC_CREATE_IMAGE]

    def __init__(self, embedding_model: AzureAIEmbedding, chatcomplete_model: AzureAIChatComplete, imagegen_model: StabilityAIImageGen, search_client: SearchClient):
        """
        Initializes a ChatBot instance.
        """
        self.embedding_model = embedding_model
        self.chatcomplete_model = chatcomplete_model
        self.imagegen_model = imagegen_model
        self.search_client = search_client

    def chat(self, messages: List[Dict]) -> List[Dict]:
        """
        Conducts a conversation between the user and the assistant.

        Args:
            messages (list): List of previous messages exchanged in the conversation.

        Returns:
            list: Generated messages exchanged in the conversation.
        """

        try:
            # Define initial system message
            message_init = {"role": "system", "content": self.ASSISTANT.format(tools=str(self.ASSISTANT_TOOLS))}

            # Initialize list to store generated messages
            generated_messages = []

            # Request response from model based on the original messages
            response = self.chatcomplete_model.predict(messages=[message_init] + messages[-20:], temperature=0.4, max_tokens=300, stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])
            response_message = response["choices"][0]["message"]["content"]
            
            # Check if the model requested a valid function call
            try:
                # Parse function
                f_dict = json.loads(response_message) 
                function_name = f_dict["function_name"]
                function_args = f_dict["parameters"]
                method = getattr(self, function_name)
            except Exception as e:
                function_name = None
                
            if function_name is not None:
                print(f_dict)
                # Call the corresponding method with the provided arguments
                function_response = method(**function_args)
                function_response_str = str(function_response) 
                print(function_response_str)
                # For images thats enough
                if function_name == "create_image":
                    response_message = function_response_str
                # For others generate a final answer
                else:
                    # Extend conversation with function response
                    generated_messages.append({"role": "tool", "name": function_name, "content": f"{function_response_str}"})

                    # Request a new response from LLM incorporating function response
                    message_init =  {"role": "system", "content": self.ASSISTANT_FINISHER}
                    response = self.chatcomplete_model.predict(messages=[message_init] + messages[-20:] + generated_messages, temperature=0.4, max_tokens=300, stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])
                    print(response)
                    response_message = response["choices"][0]["message"]["content"]

            # Append the assistant's response to the generated messages
            generated_messages.append({"role": "assistant", "content": response_message})
        except Exception as e:
            print(e)
            generated_messages.append({"role": "assistant", "content": str(e)})

        # Return generated messages and search debug information (if applicable)
        return generated_messages


    def ask_data(self, search_queries: List[str]) -> str:
        """
        Retrieve filtered data based on search queries.

        Args:
            search_queries (list): List of search queries.
            message_id (str): The ID of the message associated with the search.

        Returns:
            str: Concatenated filtered data answers.
        """
        # Initialize lists to store filtered data answers and search results debug information
        all_data_answers = []

        # Iterate over each search query
        for search_query in search_queries:

            # Retrieve search results for the current search query
            cogs_orig_results = search_knowledgebase_single(self.search_client, self.embedding_model, search_query)
            cogs_final_results = []
            filtered_info = ""

            # Iterate over search results in batches of 5
            for k in range(0, min(20, len(cogs_orig_results)), 5):

                # Iterate over each search result within the current batch
                for i in range(0, 5):
                    context = cogs_orig_results[k + i]["content"]
                    message = {
                        "role": "user",
                        "content": self.FILTERER.format(pregunta=search_query, info_sin_filtrar=context, info_filtrada_prev=filtered_info),
                    }
                    # Request response from LLM to filter the information
                    response = self.chatcomplete_model.predict(messages=[message], max_tokens=2, temperature=0.01, stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])
                    response_message = response["choices"][0]["message"]["content"]
                    # If the response is 'SI' (yes), include the context in filtered_info and cogs_final_results
                    if response_message == "SI":
                        filtered_info += context
                        cogs_final_results.append(cogs_orig_results[k + i])

                # If filtered information is found, break out of the loop
                if len(filtered_info) > 0:
                    break

            # Append the filtered data answer to all_data_answers
            all_data_answers.append(f"{filtered_info}")

        # Concatenate all filtered data answers with newline separator and return along with search results debug information
        return "\n\n".join(all_data_answers)
    

    def create_image(self, prompt_description: str, style_preset: str) -> Image:
        """
        Retrieve filtered data based on search queries.

        Args:
            prompt_description (str): Prompt describing the image.

        Returns:
            Image: Generated PIL image.
        """
        img = self.imagegen_model.predict(prompt=prompt_description, style_preset=style_preset)
        img.save(f"/........./data/images/{prompt_description[:15]}.jpeg")
        return "Image generated"
