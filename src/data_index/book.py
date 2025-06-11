class Book:

    def __init__(
        self,
        path: str,
    ):
        """
        Initializes the Book class from the path and loads text.

        Args:
            path (str):
        """
        self.path = path

        with open(path) as f:
            self.text = f.read()

        self.paragraphs = [p for p in self.text.split("\n") if len(p) > 0]
