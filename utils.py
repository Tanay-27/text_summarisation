
import config
from transformers import BartTokenizer, BartForConditionalGeneration

def load_model_and_tokenizer():
    """Load the BART model and tokenizer."""
    tokenizer = BartTokenizer.from_pretrained(config.TOKENIZER_PATH)
    model = BartForConditionalGeneration.from_pretrained(config.MODEL_PATH)
    return tokenizer, model

def read_large_text_file(filename, chunk_size=config.CHUNK_SIZE):
    """Read the large text file in chunks and yield each chunk.
    
    Args:
        filename (str): The path to the input text file.
        chunk_size (int): The size of each chunk to read.

    Yields:
        str: A chunk of text read from the file.
    
    Raises:
        FileNotFoundError: If the specified file is not found.
    """
    try:
        with open(filename, 'r') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")

def calculate_default_lengths(large_text_length):
    """
    Calculate default max_length and min_length based on percentage.

    Args:
    large_text (str): The large text to be summarized.
    
    Returns:
    int: The calculated max_length.
    int: The calculated min_length.
    """
    max_length = int(large_text_length * config.SUMMARY_LENGTH_PERCENTAGE)
    min_length = int(max_length * 0.5)  # Set min_length to half of max_length
    return max_length, min_length


