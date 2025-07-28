# PDF-trained LLM Project - We can train multiple pdf 

This project allows you to train a language model on your own PDF documents.
<img width="308" height="605" alt="image" src="https://github.com/user-attachments/assets/86c25334-92a7-4d71-8f3f-42b25aad7542" />


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

- Train the model: `python3 src/main.py --train`

## Output
  <img width="1421" height="272" alt="image" src="https://github.com/user-attachments/assets/97c26b6a-f7c4-4050-ae80-25c00f4aa534" />

**Training complete!**

- Chat with the model: `python3 src/main.py --chat`
## Output
  
  <img width="1418" height="632" alt="image" src="https://github.com/user-attachments/assets/05dcac43-6b96-4a66-a372-07d0aa8f6bda" />

  
- Create document embeddings: `python3 src/main.py --create-embeddings`
 ## Output
  <img width="1428" height="411" alt="image" src="https://github.com/user-attachments/assets/d8924bc8-6fa1-48e5-9267-8e376dc7da0b" />

- Query document embeddings: `python3 src/main.py --query "Write the Characteristics of Pseudocode."`
## Output
  <img width="1354" height="493" alt="image" src="https://github.com/user-attachments/assets/2f335b6a-711c-4b0e-bdf5-f9bbd91d4bec" />


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
