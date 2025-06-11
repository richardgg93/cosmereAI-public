import sys
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

sys.path.append("./src")
sys.path.append("./src/models")
sys.path.append("./src/chat")

from chat.chatbot import ChatBot
from models.embedding_model import AzureAIEmbedding
from models.chatcomplete_model import AzureAIChatComplete
from models.imagegen_model import StabilityAIImageGen

endpoint = ...
token = ...
model_name = "cohere-v3-multilingual-01"

embeddings_model = AzureAIEmbedding(endpoint=endpoint, token=token, model_name=model_name)

endpoint = ...
token = ...
model_name = "Meta-Llama-3-8B-Instruct-abzad"

endpoint = ...
token = ...
model_name = "Meta-Llama-3-70B-Instruct-wcukf"


chatcomplete_model = AzureAIChatComplete(endpoint=endpoint, token=token, model_name=model_name)


endpoint = ...
# endpoint = ...
token = ...
model_name = "sd3-turbo"

imagegen_model = StabilityAIImageGen(endpoint=endpoint, token=token, model_name=model_name)


AZURE_SEARCH_SERVICE_ENDPOINT = ...
AZURE_SEARCH_ADMIN_KEY = ...
AZURE_SEARCH_INDEX_NAME = "cosmere"

cogs_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=cogs_credential,
)


if __name__ == "__main__":

    messages = []
    chatbot = ChatBot(embeddings_model, chatcomplete_model, imagegen_model, search_client)

    while True:
        text = input("Insert Message: ")
        messages.append({"role": "user", "content": text})
        generated_messages = chatbot.chat(messages)

        response_text = generated_messages[-1]["content"]
        messages += generated_messages
        print(response_text)
