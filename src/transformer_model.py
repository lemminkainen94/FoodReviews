import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification


class TransformerTBSA(nn.Module):
    """
    A transformer model for Target Based Sentiment Analysis.
    You can experiment with whatever Huggingface model you feel like.
    """

    def __init__(self, args):
        super().__init__()

        self.transformer = None

        if "absa" in args.model_name:
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                args.model_name
            )
        else:
            self.transformer = AutoModel.from_pretrained(
                args.model_name, return_dict=False
            )

        self.drop = nn.Dropout(p=args.final_dropout)
        self.out = nn.Linear(self.transformer.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        try:
            pooled_output = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0][:, 0, :]
            output = self.drop(pooled_output)
            return self.out(output)
        except IndexError:
            return self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits


if __name__ == "__main__":
    pass
