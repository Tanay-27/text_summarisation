import os
import argparse
from utils import load_model_and_tokenizer, read_large_text_file
from summarization import summarize_text_chunks,extract_topics_chunks

def main():
    parser = argparse.ArgumentParser(description="Extract topics or summarize a large text file using BART.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output.")
    parser.add_argument("--task", choices=["topic_extraction", "summarization"], required=True, help="Task to perform: 'topic_extraction' or 'summarization'.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length for processing.")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum length for processing.")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk Size for processing.")
    args = parser.parse_args()

    # Load model and tokenizer
    try:
        print("Loading Models...")
        tokenizer, model = load_model_and_tokenizer()
        print("Loaded Models")
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return

    # Read the large text file
    try:
        print(f"Reading {args.input}...")
        chunks = read_large_text_file(args.input,args.chunk_size)
        print(f"Read File {args.input} Successfully")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

# Handle task: topic extraction or summarization
    if args.task == "topic_extraction":
        print("Starting Topic Extraction...")
        topics = extract_topics_chunks(chunks, tokenizer, model)
        print("Writing Topics to file...")

        # Check if the output directory exists, create it if not
        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write topics to the output file
        with open(args.output, 'w') as file:
            file.write('\n'.join(topics))
        print('Topics extracted successfully.')

    elif args.task == "summarization":
        print("Starting Summarization...")
        final_summary = summarize_text_chunks(chunks, tokenizer, model, args.max_length, args.min_length)
        print("Writing Summary to file...")

        # Check if the output directory exists, create it if not
        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write summary to the output file
        with open(args.output, 'w') as file:
            file.write(final_summary)
        print('Summary saved successfully.')

if __name__ == "__main__":
    main()