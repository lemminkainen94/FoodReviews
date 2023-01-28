<<<<<<< HEAD
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
=======
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from time import time
from torch import nn, optim
>>>>>>> main


def train_epoch(args):
    args.model = args.model.train()
    losses = []
    avg_losses = []
    temp_preds = []
    temp_targets = []
    avg_accs = []
    acc_losses = []
    correct_predictions = 0
    i = 0
    t0 = time()
    for d in args.train_dl:
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
        loss = loss / args.acc_steps
        correct_predictions += torch.sum(preds == targets)

        temp_preds += preds.cpu().tolist()
        temp_targets += targets.cpu().tolist()

        acc_losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)

        i += 1
        if i % args.acc_steps == 0:
            losses.append(np.mean(acc_losses))
            acc_losses = []
            args.optimizer.step()
<<<<<<< HEAD
            if "scheduler" in args:
=======
            if 'scheduler' in args:
>>>>>>> main
                args.scheduler.step()
            args.optimizer.zero_grad()
        if i % (100 * args.acc_steps) == 0:
            acc = 0
            try:
                acc = accuracy_score(temp_targets, temp_preds)
            except ValueError:
                pass
            temp_preds = []
            temp_targets = []

            avg_accs.append(acc)
<<<<<<< HEAD
            avg_losses.append(np.mean(losses[i - 100 : i]))
            print(
                i,
                "iters, accuracy, loss, time : ",
                avg_accs[-1],
                avg_losses[-1],
                time() - t0,
            )

    return (
        correct_predictions.double() / args.train_size,
        np.mean(losses),
        avg_losses,
        avg_accs,
    )


if __name__ == "__main__":
=======
            avg_losses.append(np.mean(losses[i-100:i]))
            print(i, 'iters, accuracy, loss, time : ', avg_accs[-1], avg_losses[-1], time()-t0)

    return correct_predictions.double() / args.train_size, np.mean(losses), avg_losses, avg_accs


if __name__ == '__main__':
>>>>>>> main
    pass
