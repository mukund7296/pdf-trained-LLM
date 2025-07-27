from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path="models/custom_llm"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        response = generate_text(prompt, model, tokenizer)
        print("\nModel response:")
        print(response)
        print("\n" + "="*50 + "\n")