import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from data_utils import train_entities, entity2idx, train_loader, test_loader
from model import model, optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute Weighted Loss for NER
num_labels = len(entity2idx)
label_counts = torch.zeros(num_labels)

for labels in train_entities:
    for label in labels:
        label_counts[label] += 1  # Count occurrences

# Compute class weights (inverse frequency)
class_weights = 1.0 / (label_counts + 1e-5)
class_weights /= class_weights.sum()

# Apply weighted loss
criterion_intent = nn.BCEWithLogitsLoss()
criterion_ner = nn.CrossEntropyLoss(weight=class_weights.to(device))


# Compute Metrics Function
def compute_metrics(intent_logits, intent_labels, ner_logits, entity_labels):
    intent_preds = (torch.sigmoid(intent_logits) > 0.5).cpu().numpy()
    intent_true = intent_labels.cpu().numpy()
    intent_acc = accuracy_score(intent_true, intent_preds)
    intent_f1 = f1_score(intent_true, intent_preds, average='micro')

    ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy().flatten()
    ner_true = entity_labels.cpu().numpy().flatten()
    ner_acc = accuracy_score(ner_true, ner_preds)
    ner_f1 = f1_score(ner_true, ner_preds, average='macro')

    return intent_acc, intent_f1, ner_acc, ner_f1

# Directory for saving models
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
checkpoint_path_template = os.path.join(checkpoint_dir, "checkpoint_epoch_{}.pt")

# Training & Evaluation Loop with Saving
num_epochs = 3
best_metric = -float('inf')  # Track best combined Intent F1 + NER F1

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_loss = 0
    intent_acc_train, intent_f1_train, ner_acc_train, ner_f1_train = [], [], [], []

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for batch in train_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent"].to(device)
        entity_labels = batch["entity_labels"].to(device)

        optimizer.zero_grad()
        intent_logits, ner_logits = model(input_ids, attention_mask)

        # Compute loss
        loss_intent = criterion_intent(intent_logits, intent_labels)
        loss_ner = criterion_ner(ner_logits.view(-1, len(entity2idx)), entity_labels.view(-1))
        loss = loss_intent + loss_ner

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute training metrics
        i_acc, i_f1, n_acc, n_f1 = compute_metrics(intent_logits, intent_labels, ner_logits, entity_labels)
        intent_acc_train.append(i_acc)
        intent_f1_train.append(i_f1)
        ner_acc_train.append(n_acc)
        ner_f1_train.append(n_f1)

        train_bar.set_postfix({
            "Loss": f"{total_loss / (train_bar.n + 1):.4f}",
            "Intent F1": f"{np.mean(intent_f1_train):.4f}",
            "NER F1": f"{np.mean(ner_f1_train):.4f}"
        })

    # Evaluation Phase
    model.eval()
    eval_loss = 0
    intent_acc_eval, intent_f1_eval, ner_acc_eval, ner_f1_eval = [], [], [], []

    eval_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eval]", leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent"].to(device)
            entity_labels = batch["entity_labels"].to(device)

            intent_logits, ner_logits = model(input_ids, attention_mask)

            loss_intent = criterion_intent(intent_logits, intent_labels)
            loss_ner = criterion_ner(ner_logits.view(-1, len(entity2idx)), entity_labels.view(-1))
            loss = loss_intent + loss_ner
            eval_loss += loss.item()

            i_acc, i_f1, n_acc, n_f1 = compute_metrics(intent_logits, intent_labels, ner_logits, entity_labels)
            intent_acc_eval.append(i_acc)
            intent_f1_eval.append(i_f1)
            ner_acc_eval.append(n_acc)
            ner_f1_eval.append(n_f1)

            eval_bar.set_postfix({
                "Loss": f"{eval_loss / (eval_bar.n + 1):.4f}",
                "Intent F1": f"{np.mean(intent_f1_eval):.4f}",
                "NER F1": f"{np.mean(ner_f1_eval):.4f}"
            })

    # Compute average metrics for the epoch
    avg_intent_f1_eval = np.mean(intent_f1_eval)
    avg_ner_f1_eval = np.mean(ner_f1_eval)
    combined_metric = (avg_intent_f1_eval + avg_ner_f1_eval) / 2  # Combined metric for best model

    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {total_loss / len(train_loader):.4f}, "
          f"Intent Acc: {np.mean(intent_acc_train):.4f}, Intent F1: {np.mean(intent_f1_train):.4f}, "
          f"NER Acc: {np.mean(ner_acc_train):.4f}, NER F1: {np.mean(ner_f1_train):.4f}")
    print(f"Eval Loss: {eval_loss / len(test_loader):.4f}, "
          f"Intent Acc: {np.mean(intent_acc_eval):.4f}, Intent F1: {avg_intent_f1_eval:.4f}, "
          f"NER Acc: {np.mean(ner_acc_eval):.4f}, NER F1: {avg_ner_f1_eval:.4f}")
    
    # Save checkpoint after each epoch
    checkpoint_path = checkpoint_path_template.format(epoch + 1)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss / len(train_loader),
        'eval_loss': eval_loss / len(test_loader),
        'intent_f1': avg_intent_f1_eval,
        'ner_f1': avg_ner_f1_eval
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model based on combined metric
    if combined_metric > best_metric:
        best_metric = combined_metric
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'intent_f1': avg_intent_f1_eval,
            'ner_f1': avg_ner_f1_eval
        }, best_model_path)
        print(f"Saved best model: {best_model_path} (Combined F1: {combined_metric:.4f})")

    print("-" * 80)