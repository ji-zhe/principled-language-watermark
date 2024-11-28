import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-cased'  # Change this if you want to use a different variant of BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

while True:
    # Input sentence with [MASK] to complete
    sentence = input("input a sentence, specital token: [MASK]. ")

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    masked_index = tokens.index("[MASK]")

    # Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create input tensors
    inputs = torch.tensor([token_ids])

    # Predict the masked token
    with torch.no_grad():
        outputs = model(inputs)
    predictions = outputs.logits[0, masked_index]

    # Get the top 5 predictions
    top_k = 5
    top_predictions = torch.topk(predictions, k=top_k, dim=0).indices.tolist()

    # Convert predicted token IDs back to tokens
    predicted_tokens = [tokenizer.convert_ids_to_tokens(pred_id) for pred_id in top_predictions]

    # Print the top predictions
    print("Original sentence:", sentence)
    print("Predicted completions:")
    for token in predicted_tokens:
        tokens[masked_index] = token
        completed_sentence = tokenizer.convert_tokens_to_string(tokens)
        print(completed_sentence)
