#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class TargetSetEncoderConfig:
    input_dim: int = 2560
    hidden_dim: int = 768
    protein_queries: int = 128
    set_queries: int = 32
    protein_layers: int = 4
    set_layers: int = 4
    num_heads: int = 12
    ffn_dim: int | None = None
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.protein_queries <= 0:
            raise ValueError("protein_queries must be positive")
        if self.set_queries < 0:
            raise ValueError("set_queries must be non-negative")
        if self.protein_layers <= 0 or self.set_layers <= 0:
            raise ValueError("protein_layers and set_layers must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if self.ffn_dim is not None and self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive when provided")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TargetSetEncoderOutput:
    """Padded target-set encoding and masks.

    A True value in either mask marks a real token. Padded protein-token rows in
    both hidden-state tensors are zero.
    """

    last_hidden_state: torch.Tensor
    attention_mask: torch.BoolTensor
    summary_hidden_state: torch.Tensor
    protein_hidden_state: torch.Tensor
    protein_attention_mask: torch.BoolTensor
    target_counts: torch.LongTensor


class ProteinCrossAttentionBlock(nn.Module):
    def __init__(self, config: TargetSetEncoderConfig) -> None:
        super().__init__()
        ffn_dim = config.ffn_dim or 4 * config.hidden_dim

        self.query_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        self.memory_norm = nn.LayerNorm(
            config.input_dim,
            eps=config.layer_norm_epsilon,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=config.input_dim,
            vdim=config.input_dim,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        residue_hidden_states: torch.Tensor,
        residue_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(residue_hidden_states)
        attention_output, _ = self.cross_attention(
            query=self.query_norm(latents),
            key=normalized_memory,
            value=normalized_memory,
            key_padding_mask=~residue_attention_mask,
            need_weights=False,
        )
        latents = latents + self.attention_dropout(attention_output)
        return latents + self.ffn(self.ffn_norm(latents))


class ProteinLatentResampler(nn.Module):
    """Compress every variable-length protein into shared fixed-length latents."""

    def __init__(self, config: TargetSetEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(
            torch.empty(config.protein_queries, config.hidden_dim)
        )
        self.layers = nn.ModuleList(
            ProteinCrossAttentionBlock(config)
            for _ in range(config.protein_layers)
        )
        self.output_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        residue_hidden_states: torch.Tensor,
        residue_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if residue_hidden_states.ndim != 3:
            raise ValueError("residue_hidden_states must have shape [P, L, input_dim]")
        if residue_hidden_states.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected residue dimension {self.config.input_dim}, "
                f"got {residue_hidden_states.size(-1)}"
            )
        if residue_attention_mask.shape != residue_hidden_states.shape[:2]:
            raise ValueError("residue_attention_mask must have shape [P, L]")
        if not torch.all(residue_attention_mask.any(dim=1)):
            raise ValueError("Every protein must contain at least one valid residue")

        batch_size = residue_hidden_states.size(0)
        latents = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(
                latents,
                residue_hidden_states,
                residue_attention_mask,
            )
        return self.output_norm(latents)


class TargetSetSelfAttentionBlock(nn.Module):
    def __init__(self, config: TargetSetEncoderConfig) -> None:
        super().__init__()
        ffn_dim = config.ffn_dim or 4 * config.hidden_dim

        self.attention_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _zero_padding(
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        return hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
        structural_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden_states)
        attention_output, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            attn_mask=structural_attention_mask,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)
        hidden_states = self._zero_padding(hidden_states, attention_mask)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return self._zero_padding(hidden_states, attention_mask)


class TargetSetEncoder(nn.Module):
    """Encode a batch of variable-size protein target sets.

    Input format:
        target_sets[batch_index][protein_index] is a floating-point tensor or
        NumPy array with shape [residue_count, input_dim].

    Output format:
        last_hidden_state has shape
        [batch_size, set_queries + max_target_count * protein_queries, hidden_dim].
        Its matching attention_mask must be passed to downstream cross-attention.

    No protein-order positional embedding is used. Summary queries can read all
    summary and protein tokens. Protein tokens can read all protein tokens, but
    cannot read summary queries, so summary information is never written back
    into the protein-token path.
    """

    def __init__(self, config: TargetSetEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TargetSetEncoderConfig()
        self.protein_resampler = ProteinLatentResampler(self.config)
        self.summary_queries = nn.Parameter(
            torch.empty(self.config.set_queries, self.config.hidden_dim)
        )
        self.set_layers = nn.ModuleList(
            TargetSetSelfAttentionBlock(self.config)
            for _ in range(self.config.set_layers)
        )
        self.output_norm = nn.LayerNorm(
            self.config.hidden_dim,
            eps=self.config.layer_norm_epsilon,
        )
        nn.init.normal_(self.summary_queries, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.summary_queries.device

    @property
    def dtype(self) -> torch.dtype:
        return self.summary_queries.dtype

    def _prepare_batch(
        self,
        target_sets: Sequence[Sequence[Any]],
    ) -> tuple[torch.Tensor, torch.BoolTensor, list[int]]:
        if (
            isinstance(target_sets, torch.Tensor)
            or getattr(target_sets, "ndim", None) is not None
        ):
            raise TypeError(
                "target_sets must be a nested sequence: "
                "target_sets[batch_index][protein_index]"
            )

        target_sets = list(target_sets)
        if not target_sets:
            raise ValueError("target_sets must contain at least one target set")

        proteins: list[torch.Tensor] = []
        target_counts: list[int] = []
        for batch_index, target_set in enumerate(target_sets):
            if (
                isinstance(target_set, torch.Tensor)
                or getattr(target_set, "ndim", None) is not None
            ):
                raise TypeError(
                    f"target_sets[{batch_index}] must be a sequence of protein encodings"
                )

            target_set = list(target_set)
            if not target_set:
                raise ValueError(f"target_sets[{batch_index}] is empty")
            target_counts.append(len(target_set))

            for protein_index, protein in enumerate(target_set):
                if isinstance(protein, torch.Tensor):
                    tensor = protein.to(device=self.device)
                else:
                    # NumPy memmaps may be read-only; copy external arrays so
                    # PyTorch never aliases non-writable storage.
                    tensor = torch.tensor(protein, device=self.device)
                if not torch.is_floating_point(tensor):
                    raise TypeError(
                        f"target_sets[{batch_index}][{protein_index}] "
                        "must be floating point"
                    )
                if tensor.ndim != 2 or tensor.size(1) != self.config.input_dim:
                    raise ValueError(
                        f"target_sets[{batch_index}][{protein_index}] must have shape "
                        f"[L, {self.config.input_dim}], got {tuple(tensor.shape)}"
                    )
                if tensor.size(0) == 0:
                    raise ValueError(
                        f"target_sets[{batch_index}][{protein_index}] has no residues"
                    )
                proteins.append(tensor.to(dtype=self.dtype))

        lengths = torch.tensor(
            [protein.size(0) for protein in proteins],
            device=self.device,
            dtype=torch.long,
        )
        padded_proteins = pad_sequence(proteins, batch_first=True)
        residue_positions = torch.arange(
            padded_proteins.size(1),
            device=self.device,
        )
        residue_attention_mask = (
            residue_positions.unsqueeze(0) < lengths.unsqueeze(1)
        )
        return padded_proteins, residue_attention_mask, target_counts

    def _pack_target_sets(
        self,
        protein_hidden_states: torch.Tensor,
        target_counts: list[int],
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        split_sets = torch.split(protein_hidden_states, target_counts, dim=0)
        padded_sets = pad_sequence(split_sets, batch_first=True)
        batch_size, max_target_count = padded_sets.shape[:2]

        target_counts_tensor = torch.tensor(
            target_counts,
            device=self.device,
            dtype=torch.long,
        )
        target_positions = torch.arange(max_target_count, device=self.device)
        target_mask = (
            target_positions.unsqueeze(0) < target_counts_tensor.unsqueeze(1)
        )
        # Every real protein contributes exactly protein_queries valid tokens.
        protein_attention_mask = (
            target_mask.unsqueeze(-1)
            .expand(batch_size, max_target_count, self.config.protein_queries)
            .reshape(batch_size, max_target_count * self.config.protein_queries)
        )
        packed_hidden_states = padded_sets.reshape(
            batch_size,
            max_target_count * self.config.protein_queries,
            self.config.hidden_dim,
        )
        packed_hidden_states = packed_hidden_states.masked_fill(
            ~protein_attention_mask.unsqueeze(-1),
            0.0,
        )
        return packed_hidden_states, protein_attention_mask, target_counts_tensor

    def _build_structural_attention_mask(
        self,
        sequence_length: int,
    ) -> torch.BoolTensor:
        mask = torch.zeros(
            sequence_length,
            sequence_length,
            device=self.device,
            dtype=torch.bool,
        )
        if self.config.set_queries:
            # PyTorch bool attn_mask uses True for blocked entries. Rows are
            # receivers and columns are sources, so this blocks protein <- summary
            # while preserving summary <- protein and protein <- protein.
            mask[self.config.set_queries :, : self.config.set_queries] = True
        return mask

    def _encode_packed_target_sets(
        self,
        protein_hidden_states: torch.Tensor,
        protein_attention_mask: torch.BoolTensor,
        target_counts_tensor: torch.LongTensor,
    ) -> TargetSetEncoderOutput:
        batch_size = protein_hidden_states.size(0)
        summary_hidden_states = self.summary_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        hidden_states = torch.cat(
            (summary_hidden_states, protein_hidden_states),
            dim=1,
        )
        summary_attention_mask = torch.ones(
            batch_size,
            self.config.set_queries,
            device=self.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat(
            (summary_attention_mask, protein_attention_mask),
            dim=1,
        )
        structural_attention_mask = self._build_structural_attention_mask(
            hidden_states.size(1)
        )

        for layer in self.set_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                structural_attention_mask,
            )
        hidden_states = self.output_norm(hidden_states)
        hidden_states = hidden_states.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )

        summary_end = self.config.set_queries
        return TargetSetEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
            summary_hidden_state=hidden_states[:, :summary_end],
            protein_hidden_state=hidden_states[:, summary_end:],
            protein_attention_mask=protein_attention_mask,
            target_counts=target_counts_tensor,
        )

    def forward(
        self,
        target_sets: Sequence[Sequence[Any]],
    ) -> TargetSetEncoderOutput:
        residue_hidden_states, residue_attention_mask, target_counts = (
            self._prepare_batch(target_sets)
        )
        protein_latents = self.protein_resampler(
            residue_hidden_states,
            residue_attention_mask,
        )
        protein_hidden_states, protein_attention_mask, target_counts_tensor = (
            self._pack_target_sets(protein_latents, target_counts)
        )
        return self._encode_packed_target_sets(
            protein_hidden_states,
            protein_attention_mask,
            target_counts_tensor,
        )

    def forward_unique(
        self,
        unique_protein_encodings: Sequence[Any],
        sample_target_indices: Sequence[Sequence[int]],
    ) -> TargetSetEncoderOutput:
        """Encode each unique protein once and reuse its latents across samples.

        sample_target_indices contains positions into unique_protein_encodings.
        Repeated indices across samples reuse the same differentiable latent
        tensor, so gradients from every occurrence accumulate at the resampler.
        """
        unique_protein_encodings = list(unique_protein_encodings)
        if not unique_protein_encodings:
            raise ValueError("unique_protein_encodings must not be empty")

        residue_hidden_states, residue_attention_mask, _ = self._prepare_batch(
            [[protein] for protein in unique_protein_encodings]
        )
        unique_protein_latents = self.protein_resampler(
            residue_hidden_states,
            residue_attention_mask,
        )

        sample_target_indices = list(sample_target_indices)
        if not sample_target_indices:
            raise ValueError("sample_target_indices must not be empty")

        gathered_target_sets: list[torch.Tensor] = []
        target_counts: list[int] = []
        unique_count = len(unique_protein_encodings)
        for sample_index, target_indices in enumerate(sample_target_indices):
            target_indices = list(target_indices)
            if not target_indices:
                raise ValueError(
                    f"sample_target_indices[{sample_index}] must not be empty"
                )
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in target_indices
            ):
                raise TypeError(
                    f"sample_target_indices[{sample_index}] must contain integers"
                )
            if len(set(target_indices)) != len(target_indices):
                raise ValueError(
                    f"sample_target_indices[{sample_index}] contains duplicates"
                )
            if any(index < 0 or index >= unique_count for index in target_indices):
                raise IndexError(
                    f"sample_target_indices[{sample_index}] contains an "
                    "out-of-range protein index"
                )

            index_tensor = torch.tensor(
                target_indices,
                device=self.device,
                dtype=torch.long,
            )
            gathered_target_sets.append(
                unique_protein_latents.index_select(0, index_tensor)
            )
            target_counts.append(len(target_indices))

        gathered_protein_latents = torch.cat(gathered_target_sets, dim=0)
        protein_hidden_states, protein_attention_mask, target_counts_tensor = (
            self._pack_target_sets(gathered_protein_latents, target_counts)
        )
        return self._encode_packed_target_sets(
            protein_hidden_states,
            protein_attention_mask,
            target_counts_tensor,
        )
