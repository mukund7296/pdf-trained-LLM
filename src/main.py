import argparse
from pdf_processor import process_pdfs
from train_model import train_model, prepare_training_data
from inference import load_model, generate_text
from embeddings import create_embeddings, load_embeddings, find_similar_texts


def main():
    parser = argparse.ArgumentParser(description="PDF-trained LLM")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--chat", action="store_true", help="Chat with the model")
    parser.add_argument("--create-embeddings", action="store_true", help="Create document embeddings")
    parser.add_argument("--query", type=str, help="Query the document embeddings")

    args = parser.parse_args()

    if args.train:
        print("Processing PDFs...")
        chunks = process_pdfs()
        if chunks:
            print("Preparing training data...")
            dataset = prepare_training_data(chunks)
            print("Training model...")
            train_model(dataset)
            print("Training complete!")

    elif args.chat:
        print("Loading model...")
        model, tokenizer = load_model()
        print("Model loaded. Start chatting (type 'quit' to exit)")
        while True:
            prompt = input("You: ")
            if prompt.lower() == 'quit':
                break
            response = generate_text(prompt, model, tokenizer)
            print("AI:", response)

    elif args.create_embeddings:
        print("Creating document embeddings...")
        chunks = process_pdfs()
        if chunks:
            create_embeddings(chunks)
            print("Embeddings created!")

    elif args.query:
        print("Loading embeddings...")
        embeddings, texts, model = load_embeddings()
        print("Most relevant texts:")
        results = find_similar_texts(args.query, embeddings, texts, model)
        for i, text in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(text)

    else:
        print("Please specify an action: --train, --chat, --create-embeddings, or --query")


if __name__ == "__main__":
    main()