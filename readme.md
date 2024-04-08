# Knowledge Extraction and Summarization

This repository contains Python scripts for extracting topics, summarizing text, and generating questions using transformer-based models. The scripts are designed to handle large files efficiently, processing them in chunks to minimize memory usage. Key features include:

- Summarization of text using transformer-based models
- Topic extraction from large text files

## Installation

To use the scripts in this repository, follow these steps:

1. Clone the repository to your local machine:
[`git clone https://github.com/Tanay-27/text_summarisation.git`](https://github.com/Tanay-27/text_summarisation.git)

2. Navigate to the cloned repository:
```cd knowledge-extraction-and-summarization```

3. Create a virtual environment (optional but recommended):
```python -m venv venv```

4. Activate the virtual environment:

- For Windows:
  ```venv\Scripts\activate```
- For macOS and Linux:
  ```source venv/bin/activate```

5. Install the required Python packages:
```pip install -r requirements.txt```


## Usage

To use the scripts for topic extraction, summarization, and question generation, follow the instructions provided in the respective script files. The scripts are designed to be modular and customizable, allowing you to adapt them to your specific use case.
This script allows you to extract topics or summarize a large text file using BART (Bidirectional and Auto-Regressive Transformer). Below are the available options and their descriptions.

## Arguments

| Argument       | Description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| `--input`      | **Required**. Path to the input text file.                                                       |
| `--output`     | **Required**. Path to save the output (extracted topics or summary).                             |
| `--task`       | **Required**. Task to perform. Choose between `topic_extraction` or `summarization`.             |
| `--max_length` | Maximum length for processing. Default is `None`.                                                |
| `--min_length` | Minimum length for processing. Default is `None`.                                                |
| `--chunk_size` | Chunk Size for processing. Default is `None`.                                                     |


## Examples

Here are some examples of how to use the scripts:

1. Summarize a large text file:
```python run.py --input "input.txt" --output "summary.txt" --task summarization --max_length 500 --min_length 100 --chunk_size 1024```
Working Example:```python run.py --input "input/BART.txt" --output "output/bartsummary.txt" --task summarization```  

2. Extract topics from a large text file:
```python run.py --input "input.txt" --output "topics.txt" --task topic_extraction```


## Contributing

Contributions to this repository are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


