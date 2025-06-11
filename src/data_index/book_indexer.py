import os
import pandas as pd
import re

import sys

sys.path.append("..")
sys.path.append(".")

from models.embedding_model import AzureAIEmbedding
from chunker import Chunker
from book import Book


class BookIndexer:

    def __init__(
        self,
        orig_data_path: str,
        chunker: Chunker,
        embedding_model: AzureAIEmbedding,
    ):
        self.orig_data_path = orig_data_path
        self.embedding_model = embedding_model
        self.chunker = chunker
        self.files_mapping = []

        self.index_column_names = [
            "id",
            "document",
            "content",
            "path",
            "vector",
        ]

    def load_data(self):
        """Loads all files"""
        for root, dir, filenames in os.walk(self.orig_data_path):
            for fname in filenames:

                # Get file information
                extension = os.path.splitext(fname)[-1]
                path_file = os.path.join(root, fname)

                # Read documents and store data
                if extension == ".txt":
                    self.files_mapping.append(
                        {
                            "file": fname,
                            "dir": dir,
                            "root": root,
                            "path_file": path_file,
                        }
                    )

        return self.files_mapping

    def generate_chunks(self):
        """Generates chunks for all files"""
        for fmap in self.files_mapping:
            book = Book(fmap["path_file"])
            chunks = self.chunk_single_book(book)
            fmap["chunks"] = chunks
        return self.files_mapping

    def chunk_single_book(self, book: Book):
        """Generate chunks for a book"""
        chunks = self.chunker.split_paragraphs(book.paragraphs)
        return chunks

    def _create_df_chunks(self, chunks, document, path):
        """Creates a DF with the extra metadata and the embeddigns for a single file"""
        # Convert to DF
        df_chunks = pd.DataFrame(chunks)

        # Generate additional data
        df_chunks["id"] = [re.sub(r"\W+", "", f"{document}_{i}") for i in range(len(df_chunks))]
        df_chunks["document"] = document
        df_chunks["path"] = path

        # Generate embeddings
        df_chunks["vector"] = self.embedding_model.predict(df_chunks["content"].to_list()).tolist()

        # Filter columns and retrieve final data only
        df_chunks = df_chunks[self.index_column_names]

        return df_chunks

    def generate_final_data(self):
        """Generates final data for all pdf files, including the embeddings and additional metadata."""
        # For every file
        all_data = []
        for fmap in self.files_mapping:
            document = fmap["file"]
            path = fmap["path_file"]
            chunks = fmap["chunks"]

            # Create DF from chunks of the file, including the embeddings
            df_chunks = self._create_df_chunks(chunks, document, path)
            all_data.append(df_chunks)

        # Concat all DFs and convert to dict
        self.final_data_index = pd.concat(all_data).to_dict(orient="records")

        return self.final_data_index

    def create_index_data(self):
        """Pipeline to generate the final data from the path.
        It follows all the steps:
        - Load data
        - Generate chunks for all book txt files (one after each other)
        - Generate the embeddings and extra metadata
        """
        self.load_data()
        self.generate_chunks()
        self.generate_final_data()
