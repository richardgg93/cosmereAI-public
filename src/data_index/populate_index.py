import sys
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

sys.path.append("..")
sys.path.append(".")


with open("/......../data/generated/final_data_index.json", "r") as f:
    final_data_index = json.load(f)

# CREDENTIALS AZURE COGNITIVE SEARCH
AZURE_SEARCH_SERVICE_ENDPOINT = ...
AZURE_SEARCH_ADMIN_KEY = ...
AZURE_SEARCH_INDEX_NAME = "cosmere"

# Create search index client
cogs_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=cogs_credential)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=cogs_credential,
)

# Upload documents to the search index
for i in range(0, len(final_data_index), 1000):
    result = search_client.upload_documents(documents=final_data_index[i : i + 1000])
    print(f"Uploaded {len(result)} documents")
