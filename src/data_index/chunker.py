import tiktoken


class TokenCounter:

    def __init__(
        self,
        encoding_name: str = "",
    ):
        """
        Initializes the TokenCounter class.

        Args:
            encoding_name (str):  The name of the encoding.
        """
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(self.encoding_name)

    def num_tokens_from_string(self, text: str) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            text (str): The text string.

        Returns:
            int: The number of tokens.
        """
        num_tokens = len(self.encoding.encode(text))
        return num_tokens


class Chunker:
    """
    This class represents an indexing tool for handling text data and creating search indices.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        Initializes the Index class with the provided parameters and configures the indexing settings.

        Args:
            chunk_size (int): The maximum token size for text chunks.
            chunk_overlap (int): The number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter("cl100k_base")

    def split_paragraphs(self, paragraphs: list[str]):
        """
        Split paragraphs into text chunks, keeping track of token limits.

        This function divides a list of paragraphs into text chunks based on specified conditions.
        It ensures that chunks do not exceed a maximum token size.
        Finally, it returns a list of chunks containing the text.

        Args:
            paragraphs (list): List of paragraphs.

        Returns:
            list: List of text chunks
        """
        chunks = []
        auxiliar_paragraphs = []
        n_token = 0
        for paragraph in paragraphs:

            # Calculate the number of tokens in the paragraph
            n_token_current = self.token_counter.num_tokens_from_string(text=paragraph)

            # If including the new paragraph exceeds the maximum token size, close chunk and start new
            if n_token + n_token_current >= self.chunk_size:

                if len(auxiliar_paragraphs) > 0:
                    chunk_content = "\n".join([p for p in auxiliar_paragraphs])
                    chunks.append(
                        {
                            "content": chunk_content,
                            "tokens": self.token_counter.num_tokens_from_string(chunk_content),
                        }
                    )

                    # Overlapping last paragraphs
                    new_auxiliar_paragraphs = []
                    last_paragraph = auxiliar_paragraphs.pop()
                    n_token = self.token_counter.num_tokens_from_string(last_paragraph)
                    while n_token < self.chunk_overlap and len(auxiliar_paragraphs) > 0:
                        new_auxiliar_paragraphs = [last_paragraph] + new_auxiliar_paragraphs
                        last_paragraph = auxiliar_paragraphs.pop() if auxiliar_paragraphs else False
                        new = self.token_counter.num_tokens_from_string(last_paragraph) if last_paragraph else 0
                        n_token += new
                    auxiliar_paragraphs = new_auxiliar_paragraphs

            # If the paragraph itself is too big, split it into smaller chunks
            if n_token_current >= self.chunk_size:
                splitted_paragraph = self.text_splitter.split_text(paragraph)
                for spl in splitted_paragraph:
                    chunks.append(
                        {
                            "content": spl,
                        }
                    )
            # If it fits just append the paragraph
            else:
                n_token += n_token_current
                auxiliar_paragraphs.append(paragraph)

        # If there are paragraphs left, split them into chunks and add them to the list
        if auxiliar_paragraphs:
            chunk_content = "\n".join([p for p in auxiliar_paragraphs])
            chunks.append({"content": chunk_content})

        return chunks
