"""
# Author: Yinghao Li
# Modified: September 30th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(
            instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---

        batch = self.tokenizer.pad(
            {"input_ids": tk_ids, "attention_mask": attn_masks},
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = [
            lb + [self.label_pad_token_id] * (max_length - len(lb)) for lb in lbs
        ]

        padded_labels = torch.tensor(padded_labels, dtype=torch.int64)
        print(f"Max sequence length: {max_length}")

        tk_ids = batch["input_ids"]

        tk_ids = torch.tensor(tk_ids, dtype=torch.int64)
        print(f"Padded tk_ids: {tk_ids.size()}")

        attn_masks = batch["attention_mask"]
        attn_masks = torch.tensor(attn_masks, dtype=torch.int64)
        lbs = padded_labels
        print(f"Padded labels: {padded_labels.size()}")

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
