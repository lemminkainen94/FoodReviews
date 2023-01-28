import numpy as np
<<<<<<< HEAD
import torch
from sklearn.metrics import f1_score
=======
from sklearn.metrics import accuracy_score
import torch
>>>>>>> main


def eval_model(args):
    args.model = args.model.eval()
    losses = []
    temp_preds = []
<<<<<<< HEAD
    temp_targets = []
=======
    temp_targets = []   
>>>>>>> main
    correct_predictions = 0
    with torch.no_grad():
        for d in args.eval_dl:
            input_ids = d["input_ids"].to(args.device)
            targets = d["targets"].to(args.device).view(-1)
            attention_mask = d["attention_mask"].to(args.device)
<<<<<<< HEAD
            outputs = args.model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.argmax(1, keepdim=True).view(-1)
=======
            outputs = args.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = outputs.argmax(1, keepdim = True).view(-1)
>>>>>>> main
            loss = args.loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            temp_preds += preds.cpu().tolist()
            temp_targets += targets.cpu().tolist()

<<<<<<< HEAD
    f1_mac = 0
    try:
        f1_mac = f1_score(temp_targets, temp_preds, average="macro")
    except ValueError:
        pass

    return f1_mac, np.mean(losses)
=======
    acc = 0
    try:
        acc = accuracy_score(temp_targets, temp_preds)
    except ValueError:
        pass

    return correct_predictions.double() / args.eval_size, np.mean(losses)
>>>>>>> main


def get_predictions(args):
    args.model = args.model.eval()
    review_texts = []
    predictions = []
<<<<<<< HEAD
=======
    prediction_probs = []
>>>>>>> main
    real_values = []
    with torch.no_grad():
        for d in args.test_dl:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(args.device)
            attention_mask = d["attention_mask"].to(args.device)
            targets = d["targets"].to(args.device)

            attention_mask = d["attention_mask"].to(args.device)
<<<<<<< HEAD
            outputs = args.model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.argmax(1, keepdim=True).view(-1)
=======
            outputs = args.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = outputs.argmax(1, keepdim = True).view(-1)
>>>>>>> main
            review_texts.extend(texts)
            predictions.extend(preds)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, real_values


<<<<<<< HEAD
def pred(args):
    args.model = args.model.eval()
    predictions = []
    with torch.no_grad():
        for d in args.dl:
            input_ids = d["input_ids"].to(args.device)
            attention_mask = d["attention_mask"].to(args.device)

            outputs = args.model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.argmax(1, keepdim=True).view(-1)
            predictions.extend(preds)

    predictions = torch.stack(predictions).cpu()
    # since the model outputs 0 for negative and the service expects -1 for negative
    return predictions - 1


if __name__ == "__main__":
    pass
=======
if __name__ == '__main__':
    pass
>>>>>>> main
