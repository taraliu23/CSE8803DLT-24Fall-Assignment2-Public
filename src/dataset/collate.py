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

        # We will use `self.tokenizer.pad()` to pad the token IDs and attention masks
        # The `padding` argument ensures that we pad to the longest sequence in the batch

        batch = self.tokenizer.pad(
            {"input_ids": tk_ids, "attention_mask": attn_masks},
            padding=True,              # Pads all sequences to the longest one
            return_tensors="pt",       # Returns PyTorch tensors
        )

        # Convert labels to tensors and pad them manually to the same length
        # Get the max length from padded token IDs
        max_length = batch["input_ids"].shape[1]
        padded_labels = [
            lb + [self.label_pad_token_id] * (max_length - len(lb)) for lb in lbs
        ]

        # Convert the padded labels to a tensor
        padded_labels = torch.tensor(padded_labels, dtype=torch.int64)

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
