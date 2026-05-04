import os
import tempfile
from pathlib import Path
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import (
    TextLoader
)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():

    # Create Temporary text file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a simple text file. \nThe file is used to demonstrate the TextLoader.")
        temp_file_path = temp_file.name


    try:
        # Load text file in TextLoader
        file_path = str(Path().cwd()) + "/demo.txt"
        loader = TextLoader(file_path)
        # loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("Document Path")
            print(temp_file_path)
            print("Document Content:")
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()
