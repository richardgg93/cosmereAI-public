# Import required libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
)


# CREDENTIALS AZURE COGNITIVE SEARCH
AZURE_SEARCH_SERVICE_ENDPOINT = ...
AZURE_SEARCH_ADMIN_KEY = ...
AZURE_SEARCH_INDEX_NAME = "cosmere"

# Create a search index
cogs_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=cogs_credential)

# Index fields definition
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchField(
        name="document",
        type=SearchFieldDataType.String,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    SearchField(
        name="content",
        type=SearchFieldDataType.String,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    SearchField(
        name="path",
        type=SearchFieldDataType.String,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1024,
        vector_search_profile_name="my-vector-config",
    ),
]

vector_search = VectorSearch(
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="my-algorithms-config",
        )
    ],
    algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
)

# Create the search index with the vector search configuration
index = SearchIndex(name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
result = index_client.create_or_update_index(index)
print(f"{result.name} created")
