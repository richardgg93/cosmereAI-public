
import sys
sys.path.append("..")
sys.path.append(".")
import json

from chunker import Chunker
from models.embedding_model import AzureAIEmbedding
from book_indexer import BookIndexer

chunker = Chunker()

endpoint = ...
token = ...
model_name = "cohere-v3-multilingual-01"
embeddings_model = AzureAIEmbedding(endpoint=endpoint, token=token, model_name=model_name)

book_indexer = BookIndexer("/......./data/books", chunker, embeddings_model)

book_indexer.load_data()
book_indexer.generate_chunks()
book_indexer.generate_final_data()

final_data_index = book_indexer.final_data_index

with open("/......../data/generated/final_data_index.json", "w") as f:
    json.dump(final_data_index, f)

