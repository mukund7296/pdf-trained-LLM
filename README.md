# PDF-trained LLM Project

This project allows you to train a language model on your own PDF documents.

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install requirements: `pip install -r requirements.txt`

## Usage

1. Add your PDFs to the `data/pdfs` directory
2. Choose one of these options:

- Train the model: `python src/main.py --train`
- Chat with the model: `python src/main.py --chat`
- Create document embeddings: `python src/main.py --create-embeddings`
- Query document embeddings: `python src/main.py --query "your question"`

## Configuration

You can modify the following parameters in the code:
- Model name (default: "gpt2")
- Training parameters (epochs, batch size, etc.)
- Chunk size for text splitting

## Deployment

To deploy this on GitHub:
1. Remove any sensitive data
2. Commit your changes: `git add . && git commit -m "Initial commit"`
3. Create a GitHub repository and push your code
# pdf-trained-LLM
