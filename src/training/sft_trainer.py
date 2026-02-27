"""
sft_trainer.py — Supervised Fine-Tuning (SFT) Trainer
=======================================================
Theory
------
SFT teaches a model to follow instructions by maximizing the log-likelihood
of human-written responses given the corresponding instructions.

Loss (cross-entropy over response tokens only):
    L_SFT = -1/T · Σ_{t=1}^{T} log p_θ(y_t | y_{<t}, x)

Instruction tokens are *masked* (label = -100) so the model is not penalized
for the prompt — it only learns to predict the response.

This is the *first stage* in RLHF and DPO pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class SFTConfig:
    """Hyperparameters for Supervised Fine-Tuning.

    Attributes
    ----------
    learning_rate : float
        Adam learning rate.
    num_epochs : int
        Training epochs.
    batch_size : int
        Samples per gradient step.
    max_length : int
        Maximum token length for the full sequence.
    warmup_steps : int
        Linear warmup steps.
    weight_decay : float
        L2 regularisation coefficient.
    logging_steps : int
        Log loss every N steps.
    """

    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 256
    warmup_steps: int = 50
    weight_decay: float = 0.01
    logging_steps: int = 10
    seed: int = 42


class InstructionDataset(Dataset):
    """Simple dataset of (instruction, response) pairs.

    Tokenizes and creates label masks that ignore instruction tokens.

    Parameters
    ----------
    examples : list[dict]
        Each dict must have keys 'instruction' and 'response'.
    tokenizer : HuggingFace tokenizer
    max_length : int
    """

    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        instruction = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
        response = ex["response"] + self.tokenizer.eos_token

        instr_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(response, add_special_tokens=False)
        full_ids = instr_ids + resp_ids

        # Truncate
        full_ids = full_ids[: self.max_length]
        n = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)

        # Labels: -100 for instruction tokens (masked), actual ids for response
        labels = torch.full((n,), -100, dtype=torch.long)
        resp_start = min(len(instr_ids), n)
        labels[resp_start:] = input_ids[resp_start:]

        # Attention mask
        attention_mask = torch.ones(n, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        labels[i, :n] = b["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SFTTrainer:
    """Trains a causal LM on instruction-response pairs using SFT.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal LM (e.g., GPT-2).
    tokenizer
        Corresponding tokenizer.
    config : SFTConfig
    """

    def __init__(self, model: nn.Module, tokenizer, config: Optional[SFTConfig] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.loss_history: List[float] = []

    def train(self, examples: List[Dict[str, str]]) -> List[float]:
        """Run SFT training on the given instruction-response pairs.

        Parameters
        ----------
        examples : list[dict]
            Each dict: {"instruction": str, "response": str}

        Returns
        -------
        list[float]
            Per-step training losses.
        """
        torch.manual_seed(self.config.seed)

        dataset = InstructionDataset(examples, self.tokenizer, self.config.max_length)
        loader = DataLoader(dataset, batch_size=self.config.batch_size,
                            shuffle=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        device = next(self.model.parameters()).device
        self.model.train()
        losses: List[float] = []
        global_step = 0

        for epoch in range(self.config.num_epochs):
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())
                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    avg = sum(losses[-self.config.logging_steps:]) / self.config.logging_steps
                    print(f"  Step {global_step} | loss={avg:.4f}")

        self.loss_history = losses
        return losses


if __name__ == "__main__":
    print("SFTTrainer — demo (requires transformers)")
    print("See Notebook 01 for a full runnable example with distilgpt2.")
