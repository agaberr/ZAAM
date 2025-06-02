import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from data_utils import intent2idx, entity2idx, tokenizer

class BiLSTM_IntentNER(nn.Module):
    def __init__(self, num_intents, num_entities):
        super(BiLSTM_IntentNER, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.intent_classifier = nn.Linear(512, num_intents)
        self.ner_classifier = nn.Linear(512, num_entities)

    def forward(self, input_ids, attention_mask):
        embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        intent_logits = self.intent_classifier(lstm_out[:, 0, :])
        ner_logits = self.ner_classifier(lstm_out)
        return intent_logits, ner_logits

def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_IntentNER(len(intent2idx), len(entity2idx)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint

def predict_intent_and_entities(model, query, tokenizer, intent2idx, entity2idx, max_length=32):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx2intent = {v: k for k, v in intent2idx.items()}
    idx2entity = {v: k for k, v in entity2idx.items()}
    encoding = tokenizer.encode_plus(
        query,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        intent_logits, ner_logits = model(input_ids, attention_mask)
    intent_preds = (torch.sigmoid(intent_logits) > 0.5).cpu().numpy().flatten()
    predicted_intents = [idx2intent[i] for i in range(len(intent_preds)) if intent_preds[i] == 1]
    ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy().flatten()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    entity_predictions = []
    for token, label_idx in zip(tokens, ner_preds):
        label = idx2entity[label_idx]
        if label != "O":
            entity_predictions.append((token, label))
    return {
        "query": query,
        "predicted_intents": predicted_intents,
        "named_entities": entity_predictions
    }
