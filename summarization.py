
import config
from utils import calculate_default_lengths

def extract_topics_chunks(chunks, tokenizer, model):
    """Extract topics from the text in chunks.
    
    Args:
        chunks (list of str): List of text chunks to extract topics from.
        tokenizer: The BART tokenizer.
        model: The BART model.

    Returns:
        list of str: List of extracted topics.

    Raises:
        None
    """
    topics = []
    for idx,chunk in enumerate(chunks):
        print(f'Processing Chunk {idx}...')
        inputs = tokenizer.encode("extract topics: " +chunk, return_tensors="pt", max_length=config.CHUNK_SIZE, truncation=True)
        topic_ids = model.generate(inputs, max_length=config.EXTRACTED_TOPIC_MAX_LENGTH, min_length=config.EXTRACTED_TOPIC_MIN_LENGTH, length_penalty=config.DEFAULT_LENGTH_PENALTY, num_beams=config.DEFAULT_NUM_BEAMS, early_stopping=True)
        topic = tokenizer.decode(topic_ids[0], skip_special_tokens=True)
        topics.append(topic)
    return topics

def summarize_text_chunks(chunks, tokenizer, model, max_length, min_length):
    """Summarize the text in chunks.
    
    Args:
        chunks (list of str): List of text chunks to summarize.
        tokenizer: The BART tokenizer.
        model: The BART model.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The final summary of the text.

    Raises:
        None
    """
    final_summary = ""
    for idx,chunk in enumerate(chunks):
        if not max_length or not min_length: max_length,min_length = calculate_default_lengths(len(chunk))
        print(f'Processing Chunk {idx}...')
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=config.CHUNK_SIZE, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=config.DEFAULT_LENGTH_PENALTY, num_beams=config.DEFAULT_NUM_BEAMS, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary += summary + "\n"
    return final_summary
