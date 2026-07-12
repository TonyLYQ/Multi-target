#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


SMILES_TOKEN_RE = re.compile(r"(\[[^\[\]]+\]|Br|Cl|%\d{2}|.)")
DEFAULT_SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[UNK]")


class SmilesTokenizer:
    """Rule-based SMILES tokenizer backed by a CSV vocabulary."""

    def __init__(
        self,
        vocab_path: str | Path,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.token_to_id = self._load_vocab(self.vocab_path)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token, self.bos_token, self.eos_token, self.unk_token = tuple(special_tokens)
        self.pad_token_id = self._require_token(self.pad_token)
        self.bos_token_id = self._require_token(self.bos_token)
        self.eos_token_id = self._require_token(self.eos_token)
        self.unk_token_id = self._require_token(self.unk_token)

    @staticmethod
    def _load_vocab(vocab_path: Path) -> dict[str, int]:
        token_to_id: dict[str, int] = {}
        seen_ids: set[int] = set()
        with vocab_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"token_id", "token"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"Vocabulary CSV must contain columns: {sorted(required_columns)}")

            for row in reader:
                token = row["token"]
                token_id = int(row["token_id"])
                if token in token_to_id:
                    raise ValueError(f"Duplicate token in vocabulary: {token}")
                if token_id in seen_ids:
                    raise ValueError(f"Duplicate token_id in vocabulary: {token_id}")
                token_to_id[token] = token_id
                seen_ids.add(token_id)

        if not token_to_id:
            raise ValueError(f"Empty vocabulary: {vocab_path}")
        expected_ids = set(range(len(token_to_id)))
        if seen_ids != expected_ids:
            raise ValueError("Vocabulary token_id values must be contiguous and start from 0")
        return token_to_id

    def _require_token(self, token: str) -> int:
        if token not in self.token_to_id:
            raise ValueError(f"Required token {token!r} is missing from {self.vocab_path}")
        return self.token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, smiles: str) -> list[str]:
        return SMILES_TOKEN_RE.findall(smiles.strip())

    def encode(self, smiles: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens = self.tokenize(smiles)
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        tokens: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.id_to_token.get(token_id, self.unk_token))
        return "".join(tokens)


@dataclass
class SmilesGPT2ModelConfig:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    max_position_embeddings: int = 128
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    add_cross_attention: bool = False

    def to_gpt2_config(self) -> GPT2Config:
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.max_position_embeddings,
            n_ctx=self.max_position_embeddings,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.resid_pdrop,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            add_cross_attention=self.add_cross_attention,
        )


def build_smiles_gpt2_config(
    tokenizer: SmilesTokenizer,
    max_position_embeddings: int = 128,
    n_embd: int = 512,
    n_layer: int = 8,
    n_head: int = 8,
    resid_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    attn_pdrop: float = 0.1,
    layer_norm_epsilon: float = 1e-5,
    add_cross_attention: bool = False,
) -> GPT2Config:
    model_config = SmilesGPT2ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=max_position_embeddings,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        layer_norm_epsilon=layer_norm_epsilon,
        add_cross_attention=add_cross_attention,
    )
    return model_config.to_gpt2_config()


class ProteinConditionedGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 that keeps protein memory available throughout generation."""

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, object]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs["encoder_hidden_states"] = kwargs.get(
            "encoder_hidden_states"
        )
        model_inputs["encoder_attention_mask"] = kwargs.get(
            "encoder_attention_mask"
        )
        return model_inputs


class SmilesGPT2Generator(nn.Module):
    """GPT-2 SMILES generator with optional protein cross-attention."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.gpt2 = ProteinConditionedGPT2LMHeadModel(config)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: SmilesTokenizer,
        max_position_embeddings: int = 128,
        n_embd: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        add_cross_attention: bool = False,
    ) -> "SmilesGPT2Generator":
        config = build_smiles_gpt2_config(
            tokenizer=tokenizer,
            max_position_embeddings=max_position_embeddings,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            add_cross_attention=add_cross_attention,
        )
        return cls(config)

    def _validate_conditioning(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if encoder_hidden_states is None:
            if encoder_attention_mask is not None:
                raise ValueError(
                    "encoder_attention_mask requires encoder_hidden_states"
                )
            return None

        if not self.config.add_cross_attention:
            raise ValueError(
                "Protein conditioning requires add_cross_attention=True"
            )
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape [B, S, n_embd]"
            )
        if encoder_hidden_states.size(0) != input_ids.size(0):
            raise ValueError(
                "input_ids and encoder_hidden_states must have the same batch size"
            )
        if encoder_hidden_states.size(-1) != self.config.n_embd:
            raise ValueError(
                f"Expected encoder hidden dimension {self.config.n_embd}, "
                f"got {encoder_hidden_states.size(-1)}"
            )
        if encoder_hidden_states.device != input_ids.device:
            raise ValueError(
                "input_ids and encoder_hidden_states must be on the same device"
            )

        expected_mask_shape = encoder_hidden_states.shape[:2]
        if encoder_attention_mask is None:
            return torch.ones(
                expected_mask_shape,
                device=encoder_hidden_states.device,
                dtype=torch.bool,
            )
        if encoder_attention_mask.shape != expected_mask_shape:
            raise ValueError(
                "encoder_attention_mask must have shape [B, S] matching "
                "encoder_hidden_states"
            )
        return encoder_attention_mask.to(
            device=encoder_hidden_states.device,
            dtype=torch.bool,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        """Run unconditional or protein-conditioned causal language modeling."""
        encoder_attention_mask = self._validate_conditioning(
            input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        encoder_attention_mask = self._validate_conditioning(
            input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        return self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            **kwargs,
        )
