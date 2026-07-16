#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.smiles_gen import SmilesGPT2Generator, SmilesTokenizer
from model.target_set_encoder import TargetSetEncoder, TargetSetEncoderConfig
from utils.early_stopping import EarlyStopping


@dataclass(frozen=True)
class PositiveRecord:
    source_row_number: int
    smiles: str
    target_ids: tuple[str, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return config


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def print_main(message: str, rank: int) -> None:
    if is_main_process(rank):
        print(message, flush=True)


def precision_context(mixed_precision: str, device: torch.device):
    if device.type != "cuda" or mixed_precision in {"no", "none", "fp32"}:
        return torch.autocast(device_type=device.type, enabled=False)
    if mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown mixed precision mode: {mixed_precision}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(
    mixed_precision: str,
) -> torch.cuda.amp.GradScaler | None:
    if torch.cuda.is_available() and mixed_precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


def parse_target_ids(raw_target_ids: str | None, row_number: int) -> tuple[str, ...]:
    if raw_target_ids is None:
        raise ValueError(f"Row {row_number}: target_ids is missing")

    target_ids: list[str] = []
    seen: set[str] = set()
    for value in raw_target_ids.split(","):
        target_id = value.strip()
        if not target_id or target_id in seen:
            continue
        if Path(target_id).name != target_id:
            raise ValueError(f"Row {row_number}: invalid target ID {target_id!r}")
        seen.add(target_id)
        target_ids.append(target_id)
    if len(target_ids) < 2:
        raise ValueError(
            f"Row {row_number}: at least two unique target IDs are required"
        )
    return tuple(target_ids)


def read_positive_records(csv_path: Path) -> list[PositiveRecord]:
    records: list[PositiveRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"smiles", "target_ids"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Input CSV must contain columns: {sorted(required_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                raise ValueError(f"Row {row_number}: smiles is empty")
            records.append(
                PositiveRecord(
                    source_row_number=row_number,
                    smiles=smiles,
                    target_ids=parse_target_ids(
                        row.get("target_ids"),
                        row_number,
                    ),
                )
            )

    if not records:
        raise ValueError(f"No data rows found in {csv_path}")
    return records


def filter_records_with_available_proteins(
    records: Sequence[PositiveRecord],
    protein_dir: Path,
) -> tuple[
    list[PositiveRecord],
    list[tuple[PositiveRecord, tuple[str, ...]]],
]:
    available_records: list[PositiveRecord] = []
    skipped_records: list[tuple[PositiveRecord, tuple[str, ...]]] = []
    for record in records:
        missing_target_ids = tuple(
            target_id
            for target_id in record.target_ids
            if not (protein_dir / f"{target_id}.npy").is_file()
        )
        if missing_target_ids:
            skipped_records.append((record, missing_target_ids))
        else:
            available_records.append(record)
    return available_records, skipped_records


def validate_protein_files(
    records: Sequence[PositiveRecord],
    protein_dir: Path,
    input_dim: int,
) -> list[str]:
    unique_target_ids = list(
        dict.fromkeys(
            target_id
            for record in records
            for target_id in record.target_ids
        )
    )
    for target_id in unique_target_ids:
        protein_path = protein_dir / f"{target_id}.npy"
        if not protein_path.is_file():
            raise FileNotFoundError(
                f"Missing protein encoding for {target_id}: {protein_path}"
            )
        encoding = np.load(protein_path, mmap_mode="r", allow_pickle=False)
        if encoding.ndim != 2 or encoding.shape[1] != input_dim:
            raise ValueError(
                f"{protein_path} must have shape [L, {input_dim}], "
                f"got {encoding.shape}"
            )
        if encoding.shape[0] == 0:
            raise ValueError(f"Protein encoding has no residues: {protein_path}")
        if not np.issubdtype(encoding.dtype, np.floating):
            raise TypeError(
                f"Protein encoding must be floating point: {protein_path}"
            )
        del encoding
    return unique_target_ids


def fingerprint_records(records: Sequence[PositiveRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.source_row_number).encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.smiles.encode("utf-8"))
        digest.update(b"\0")
        digest.update(",".join(record.target_ids).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_split_ratio(split_ratio: Sequence[float]) -> tuple[float, float, float]:
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must contain train, valid, and test ratios")
    ratios = tuple(float(value) for value in split_ratio)
    if any(value < 0.0 for value in ratios):
        raise ValueError("split_ratio values must be non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("split_ratio values must sum to 1")
    if ratios[0] <= 0.0 or ratios[1] <= 0.0:
        raise ValueError("train and validation ratios must be positive")
    return ratios


def build_or_load_splits(
    split_path: Path,
    csv_path: Path,
    record_count: int,
    record_fingerprint: str,
    seed: int,
    split_ratio: tuple[float, float, float],
    distributed: bool,
    rank: int,
) -> dict[str, Any]:
    expected_metadata = {
        "csv_path": str(csv_path.resolve()),
        "record_count": record_count,
        "record_fingerprint": record_fingerprint,
        "seed": seed,
        "split_ratio": list(split_ratio),
    }
    should_build = False
    if is_main_process(rank):
        should_build = not split_path.exists()
        if split_path.exists():
            existing_splits = torch.load(split_path, map_location="cpu")
            if existing_splits.get("metadata", {}) != expected_metadata:
                print(
                    f"Existing split does not match the currently available "
                    f"data; rebuilding {split_path}.",
                    flush=True,
                )
                should_build = True

    if is_main_process(rank) and should_build:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(record_count, generator=generator)
        train_ratio, valid_ratio, _ = split_ratio
        train_count = int(record_count * train_ratio)
        valid_count = int(record_count * valid_ratio)
        if train_count == 0 or valid_count == 0:
            raise ValueError(
                "Dataset is too small for the requested train/validation split"
            )

        splits: dict[str, Any] = {
            "train": permutation[:train_count].clone(),
            "valid": permutation[train_count : train_count + valid_count].clone(),
            "test": permutation[train_count + valid_count :].clone(),
            "metadata": expected_metadata,
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(splits, split_path)
        print(
            f"Split index saved to {split_path}: "
            f"train={len(splits['train'])}, valid={len(splits['valid'])}, "
            f"test={len(splits['test'])}",
            flush=True,
        )

    barrier(distributed)
    splits = torch.load(split_path, map_location="cpu")
    metadata = splits.get("metadata", {})
    if metadata != expected_metadata:
        raise ValueError(
            f"Existing split metadata does not match the current data: {split_path}"
        )
    return splits


class PositiveTargetDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PositiveRecord],
        indices: torch.Tensor,
        tokenizer: SmilesTokenizer,
        max_length: int,
    ) -> None:
        self.records = records
        self.indices = indices.cpu().long().tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[list[int], tuple[str, ...]]:
        record = self.records[self.indices[index]]
        token_ids = self.tokenizer.encode(
            record.smiles,
            add_bos=True,
            add_eos=True,
        )
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
            token_ids[-1] = self.tokenizer.eos_token_id
        return token_ids, record.target_ids


class PositiveTargetCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: list[tuple[list[int], tuple[str, ...]]],
    ) -> dict[str, Any]:
        batch_size = len(batch)
        max_length = max(len(token_ids) for token_ids, _ in batch)
        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

        unique_target_ids: list[str] = []
        target_to_index: dict[str, int] = {}
        sample_target_indices: list[list[int]] = []
        for row, (token_ids, target_ids) in enumerate(batch):
            length = len(token_ids)
            input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row, :length] = 1

            target_indices: list[int] = []
            for target_id in target_ids:
                if target_id not in target_to_index:
                    target_to_index[target_id] = len(unique_target_ids)
                    unique_target_ids.append(target_id)
                target_indices.append(target_to_index[target_id])
            sample_target_indices.append(target_indices)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "unique_target_ids": unique_target_ids,
            "sample_target_indices": sample_target_indices,
        }


def make_dataloader(
    dataset: PositiveTargetDataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=PositiveTargetCollator(dataset.tokenizer.pad_token_id),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def load_unique_protein_encodings(
    unique_target_ids: Sequence[str],
    protein_dir: Path,
) -> list[np.ndarray]:
    return [
        np.load(
            protein_dir / f"{target_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for target_id in unique_target_ids
    ]


class ProteinConditionedSmilesFineTuner(nn.Module):
    def __init__(
        self,
        smiles_model: SmilesGPT2Generator,
        target_encoder: TargetSetEncoder,
    ) -> None:
        super().__init__()
        self.smiles_model = smiles_model
        self.target_encoder = target_encoder

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        unique_protein_encodings: Sequence[Any],
        sample_target_indices: Sequence[Sequence[int]],
    ):
        target_output = self.target_encoder.forward_unique(
            unique_protein_encodings,
            sample_target_indices,
        )
        return self.smiles_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=target_output.last_hidden_state,
            encoder_attention_mask=target_output.attention_mask,
        )


def load_pretrained_smiles_weights(
    smiles_model: SmilesGPT2Generator,
    checkpoint_path: Path,
) -> tuple[list[str], list[str]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Invalid pretrained checkpoint: {checkpoint_path}")

    incompatible = smiles_model.load_state_dict(state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in missing_keys
        if ".crossattention." not in key and ".ln_cross_attn." not in key
    ]
    if invalid_missing or unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint is incompatible. "
            f"Invalid missing keys: {invalid_missing}; "
            f"unexpected keys: {unexpected_keys}"
        )
    return missing_keys, unexpected_keys


def build_joint_model(
    tokenizer: SmilesTokenizer,
    pretrained_config: dict[str, Any],
    target_config: dict[str, Any],
) -> ProteinConditionedSmilesFineTuner:
    model_config = pretrained_config["model"]
    smiles_model = SmilesGPT2Generator.from_tokenizer(
        tokenizer=tokenizer,
        max_position_embeddings=int(
            model_config.get("max_position_embeddings", 128)
        ),
        n_embd=int(model_config.get("n_embd", 512)),
        n_layer=int(model_config.get("n_layer", 8)),
        n_head=int(model_config.get("n_head", 8)),
        resid_pdrop=float(model_config.get("resid_pdrop", 0.1)),
        embd_pdrop=float(model_config.get("embd_pdrop", 0.1)),
        attn_pdrop=float(model_config.get("attn_pdrop", 0.1)),
        layer_norm_epsilon=float(
            model_config.get("layer_norm_epsilon", 1e-5)
        ),
        add_cross_attention=True,
    )

    target_encoder_config = TargetSetEncoderConfig(
        input_dim=int(target_config.get("input_dim", 2560)),
        hidden_dim=int(target_config.get("hidden_dim", 768)),
        protein_queries=int(target_config.get("protein_queries", 128)),
        set_queries=int(target_config.get("set_queries", 32)),
        protein_layers=int(target_config.get("protein_layers", 4)),
        set_layers=int(target_config.get("set_layers", 4)),
        num_heads=int(target_config.get("num_heads", 12)),
        ffn_dim=(
            None
            if target_config.get("ffn_dim") is None
            else int(target_config["ffn_dim"])
        ),
        dropout=float(target_config.get("dropout", 0.1)),
        layer_norm_epsilon=float(
            target_config.get("layer_norm_epsilon", 1e-5)
        ),
    )
    if target_encoder_config.hidden_dim != smiles_model.config.n_embd:
        raise ValueError(
            "Target encoder hidden_dim must equal the pretrained GPT n_embd: "
            f"{target_encoder_config.hidden_dim} != {smiles_model.config.n_embd}"
        )
    return ProteinConditionedSmilesFineTuner(
        smiles_model=smiles_model,
        target_encoder=TargetSetEncoder(target_encoder_config),
    )


def build_optimizer(
    model: ProteinConditionedSmilesFineTuner,
    training_config: dict[str, Any],
) -> AdamW:
    base_gpt_parameters: list[nn.Parameter] = []
    cross_attention_parameters: list[nn.Parameter] = []
    for name, parameter in model.smiles_model.named_parameters():
        if ".crossattention." in name or ".ln_cross_attn." in name:
            cross_attention_parameters.append(parameter)
        else:
            base_gpt_parameters.append(parameter)

    parameter_groups = [
        {
            "name": "gpt",
            "params": base_gpt_parameters,
            "lr": float(training_config.get("gpt_learning_rate", 1e-5)),
        },
        {
            "name": "cross_attention",
            "params": cross_attention_parameters,
            "lr": float(
                training_config.get("cross_attention_learning_rate", 1e-4)
            ),
        },
        {
            "name": "target_encoder",
            "params": list(model.target_encoder.parameters()),
            "lr": float(
                training_config.get("target_encoder_learning_rate", 1e-4)
            ),
        },
    ]
    return AdamW(
        parameter_groups,
        betas=tuple(training_config.get("betas", [0.9, 0.95])),
        eps=float(training_config.get("eps", 1e-8)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary_path)
    temporary_path.replace(path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopper: EarlyStopping,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_valid_loss: float | None,
    last_train_loss: float | None,
    last_valid_loss: float | None,
    save_state_dict_only: bool,
) -> None:
    state = model_state(model)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopping_state_dict": early_stopper.state_dict(),
        "best_valid_loss": best_valid_loss,
        "last_train_loss": last_train_loss,
        "last_valid_loss": last_valid_loss,
        "config": config,
    }
    atomic_torch_save(checkpoint, path)
    if save_state_dict_only:
        state_path = path.with_name(path.stem + "_state_dict.pt")
        atomic_torch_save(state, state_path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopper: EarlyStopping,
    map_location: str | torch.device,
) -> tuple[int, int, float | None]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    early_stopper.load_state_dict(checkpoint["early_stopping_state_dict"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        checkpoint.get("best_valid_loss"),
    )


def token_count(labels: torch.LongTensor) -> torch.Tensor:
    return (labels[:, 1:] != -100).sum()


def globally_average_loss(
    loss_sum: float,
    count: int,
    device: torch.device,
    distributed: bool,
) -> float:
    totals = torch.tensor(
        [loss_sum, float(count)],
        device=device,
        dtype=torch.float64,
    )
    reduce_sum(totals, distributed)
    return float((totals[0] / totals[1].clamp_min(1.0)).item())


def log_learning_rates(
    writer: SummaryWriter,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> None:
    for parameter_group in optimizer.param_groups:
        writer.add_scalar(
            f"learning_rate/{parameter_group.get('name', 'group')}",
            parameter_group["lr"],
            global_step,
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    protein_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    mixed_precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    global_step: int,
    rank: int,
    distributed: bool,
    writer: SummaryWriter | None,
    logging_steps: int,
    checkpoint_steps: int,
    checkpoint_dir: Path,
    early_stopper: EarlyStopping,
    config: dict[str, Any],
    best_valid_loss: float | None,
    save_state_dict_only: bool,
) -> tuple[int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_loss_sum = 0.0
    epoch_token_count = 0
    running_loss_sum = 0.0
    running_token_count = 0
    last_logged_loss = math.nan

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        unique_protein_encodings = load_unique_protein_encodings(
            batch["unique_target_ids"],
            protein_dir,
        )

        group_start = ((step - 1) // gradient_accumulation_steps) * (
            gradient_accumulation_steps
        )
        steps_in_group = min(
            gradient_accumulation_steps,
            len(loader) - group_start,
        )
        with precision_context(mixed_precision, device):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                unique_protein_encodings=unique_protein_encodings,
                sample_target_indices=batch["sample_target_indices"],
            )
            loss = outputs.loss
            scaled_loss = loss / steps_in_group

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        valid_tokens = int(token_count(labels).item())
        batch_loss_sum = float(loss.detach().item()) * valid_tokens
        epoch_loss_sum += batch_loss_sum
        epoch_token_count += valid_tokens
        running_loss_sum += batch_loss_sum
        running_token_count += valid_tokens
        del unique_protein_encodings

        should_step = (
            step % gradient_accumulation_steps == 0 or step == len(loader)
        )
        if not should_step:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if logging_steps and global_step % logging_steps == 0:
            last_logged_loss = globally_average_loss(
                running_loss_sum,
                running_token_count,
                device,
                distributed,
            )
            if is_main_process(rank):
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={last_logged_loss:.6f}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", last_logged_loss, global_step)
                    writer.add_scalar(
                        "train/perplexity",
                        math.exp(min(last_logged_loss, 20.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/gradient_norm",
                        float(gradient_norm),
                        global_step,
                    )
                    log_learning_rates(writer, optimizer, global_step)
            running_loss_sum = 0.0
            running_token_count = 0

        if (
            is_main_process(rank)
            and checkpoint_steps
            and global_step % checkpoint_steps == 0
        ):
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                early_stopper,
                epoch=epoch,
                global_step=global_step,
                config=config,
                best_valid_loss=best_valid_loss,
                last_train_loss=(
                    None if math.isnan(last_logged_loss) else last_logged_loss
                ),
                last_valid_loss=None,
                save_state_dict_only=save_state_dict_only,
            )

    return global_step, globally_average_loss(
        epoch_loss_sum,
        epoch_token_count,
        device,
        distributed,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    protein_dir: Path,
    device: torch.device,
    mixed_precision: str,
    distributed: bool,
) -> float:
    model.eval()
    total_loss_sum = 0.0
    total_token_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        unique_protein_encodings = load_unique_protein_encodings(
            batch["unique_target_ids"],
            protein_dir,
        )
        with precision_context(mixed_precision, device):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                unique_protein_encodings=unique_protein_encodings,
                sample_target_indices=batch["sample_target_indices"],
            )
        valid_tokens = int(token_count(labels).item())
        total_loss_sum += float(outputs.loss.item()) * valid_tokens
        total_token_count += valid_tokens
        del unique_protein_encodings

    return globally_average_loss(
        total_loss_sum,
        total_token_count,
        device,
        distributed,
    )


def resolve_config(
    config: dict[str, Any],
    pretrained_config: dict[str, Any],
    pretrained_config_path: Path,
    pretrained_checkpoint_path: Path,
    tokenizer_path: Path,
    max_length: int,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["resolved_pretrained"] = {
        "config_path": str(pretrained_config_path),
        "checkpoint_path": str(pretrained_checkpoint_path),
        "model": copy.deepcopy(pretrained_config["model"]),
        "vocab_path": str(tokenizer_path),
    }
    resolved["resolved_pretrained"]["model"]["add_cross_attention"] = True
    resolved.setdefault("data", {})["max_length"] = max_length
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a protein-conditioned GPT-2 SMILES generator."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/smiles_gen_finetune_config.yaml"),
        help="Fine-tuning config YAML.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a full downstream checkpoint such as latest.pt.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.exists() else project_path(args.config)
    config = load_yaml(config_path)
    distributed, rank, local_rank, world_size = init_distributed()
    writer: SummaryWriter | None = None

    try:
        seed = int(config.get("seed", 42))
        set_seed(seed + rank)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        training_config = config["training"]
        output_dir = project_path(
            training_config.get("output_dir", "outputs/smiles_gen_finetune")
        )
        checkpoint_dir = output_dir / "checkpoints"
        tensorboard_dir = output_dir / "tensorboard"
        data_config = config["data"]
        split_path_value = data_config.get("split_index_path")
        split_path = (
            project_path(split_path_value)
            if split_path_value
            else output_dir / "splits" / "positive_split_indices.pt"
        )

        pretrained_section = config["pretrained"]
        pretrained_output_dir = project_path(pretrained_section["output_dir"])
        pretrained_config_path = pretrained_output_dir / str(
            pretrained_section.get("resolved_config", "resolved_config.yaml")
        )
        pretrained_checkpoint_path = pretrained_output_dir / str(
            pretrained_section.get(
                "checkpoint",
                "checkpoints/best_state_dict.pt",
            )
        )
        if not pretrained_config_path.is_file():
            raise FileNotFoundError(
                f"Pretrained resolved config not found: {pretrained_config_path}"
            )
        if not pretrained_checkpoint_path.is_file() and args.resume is None:
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_checkpoint_path}"
            )

        pretrained_config = load_yaml(pretrained_config_path)
        vocab_path_value = data_config.get("vocab_path")
        tokenizer_path = project_path(
            vocab_path_value or pretrained_config["data"]["vocab_path"]
        )
        tokenizer = SmilesTokenizer(tokenizer_path)
        max_length_value = data_config.get("max_length")
        max_length = int(
            max_length_value
            if max_length_value is not None
            else pretrained_config["data"].get("max_length", 128)
        )
        max_positions = int(
            pretrained_config["model"].get("max_position_embeddings", 128)
        )
        if max_length > max_positions:
            raise ValueError(
                f"data.max_length ({max_length}) exceeds GPT capacity "
                f"({max_positions})"
            )

        csv_path = project_path(data_config["csv_path"])
        protein_dir = project_path(data_config["protein_dir"])
        if not csv_path.is_file():
            raise FileNotFoundError(f"Positive CSV not found: {csv_path}")
        if not protein_dir.is_dir():
            raise FileNotFoundError(
                f"Protein encoding directory not found: {protein_dir}"
            )

        records = read_positive_records(csv_path)
        records, skipped_records = filter_records_with_available_proteins(
            records,
            protein_dir,
        )
        if is_main_process(rank):
            for record, missing_target_ids in skipped_records:
                missing_description = ", ".join(
                    f"{target_id} ({protein_dir / f'{target_id}.npy'})"
                    for target_id in missing_target_ids
                )
                print(
                    f"Skipping CSV row {record.source_row_number}: "
                    f"SMILES={record.smiles!r}; missing protein "
                    f"encoding(s): {missing_description}",
                    flush=True,
                )
            if skipped_records:
                print(
                    f"Skipped {len(skipped_records)} data row(s) with missing "
                    f"protein encodings; retained {len(records)} row(s).",
                    flush=True,
                )
        if not records:
            raise ValueError(
                "No usable data rows remain after removing records with missing "
                "protein encodings"
            )
        target_config = config.get("target_encoder", {})
        input_dim = int(target_config.get("input_dim", 2560))
        unique_target_ids = validate_protein_files(
            records,
            protein_dir,
            input_dim,
        )
        split_ratio = validate_split_ratio(
            data_config.get("split_ratio", [0.8, 0.1, 0.1])
        )
        splits = build_or_load_splits(
            split_path=split_path,
            csv_path=csv_path,
            record_count=len(records),
            record_fingerprint=fingerprint_records(records),
            seed=seed,
            split_ratio=split_ratio,
            distributed=distributed,
            rank=rank,
        )

        train_dataset = PositiveTargetDataset(
            records,
            splits["train"],
            tokenizer,
            max_length,
        )
        valid_dataset = PositiveTargetDataset(
            records,
            splits["valid"],
            tokenizer,
            max_length,
        )
        num_workers = int(training_config.get("num_workers", 0))
        train_batch_size = int(
            training_config.get("per_device_train_batch_size", 32)
        )
        eval_batch_size = int(
            training_config.get("per_device_eval_batch_size", 32)
        )
        if train_batch_size <= 0 or eval_batch_size <= 0:
            raise ValueError("Training and evaluation batch sizes must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        train_loader, train_sampler = make_dataloader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            distributed=distributed,
            shuffle=True,
        )
        valid_loader, valid_sampler = make_dataloader(
            valid_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            distributed=distributed,
            shuffle=False,
        )
        if len(train_loader) == 0 or len(valid_loader) == 0:
            raise ValueError("Training and validation loaders must not be empty")

        model = build_joint_model(
            tokenizer,
            pretrained_config,
            target_config,
        )
        if args.resume is None:
            missing_keys, _ = load_pretrained_smiles_weights(
                model.smiles_model,
                pretrained_checkpoint_path,
            )
            print_main(
                f"Loaded pretrained GPT; initialized {len(missing_keys)} "
                "new cross-attention parameters.",
                rank,
            )
        model = model.to(device)

        optimizer = build_optimizer(model, training_config)
        max_epochs = int(training_config.get("max_epochs", 100))
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        gradient_accumulation_steps = int(
            training_config.get("gradient_accumulation_steps", 1)
        )
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        updates_per_epoch = math.ceil(
            len(train_loader) / gradient_accumulation_steps
        )
        total_steps = max_epochs * updates_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                total_steps * float(training_config.get("warmup_ratio", 0.05))
            ),
            num_training_steps=max(total_steps, 1),
        )
        early_config = config.get("early_stopping", {})
        early_stopper = EarlyStopping(
            patience=int(early_config.get("patience", 10)),
            min_delta=float(early_config.get("min_delta", 0.0)),
            mode="min",
        )

        start_epoch = 0
        global_step = 0
        best_valid_loss = None
        if args.resume is not None:
            resume_path = (
                args.resume if args.resume.exists() else project_path(args.resume)
            )
            start_epoch, global_step, best_valid_loss = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                early_stopper,
                map_location=device,
            )
            print_main(
                f"Resumed from {resume_path} at epoch={start_epoch}, "
                f"step={global_step}.",
                rank,
            )

        resolved_config = resolve_config(
            config,
            pretrained_config,
            pretrained_config_path,
            pretrained_checkpoint_path,
            tokenizer_path,
            max_length,
        )
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "resolved_config.yaml").open(
                "w",
                encoding="utf-8",
            ) as handle:
                yaml.safe_dump(resolved_config, handle, sort_keys=False)
        barrier(distributed)

        print_main(
            f"World size: {world_size}, device: {device}, records: {len(records)}, "
            f"unique proteins: {len(unique_target_ids)}.",
            rank,
        )
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )

        mixed_precision = str(
            training_config.get("mixed_precision", "bf16")
        ).lower()
        if mixed_precision not in {"no", "none", "fp32", "fp16", "bf16"}:
            raise ValueError(
                "mixed_precision must be one of: no, none, fp32, fp16, bf16"
            )
        if (
            mixed_precision == "bf16"
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            mixed_precision = "fp16"
            print_main(
                "bf16 is unsupported on this GPU; falling back to fp16.",
                rank,
            )
        scaler = make_grad_scaler(mixed_precision)
        writer = (
            SummaryWriter(str(tensorboard_dir))
            if is_main_process(rank)
            else None
        )
        logging_steps = int(training_config.get("logging_steps", 1))
        checkpoint_steps = int(training_config.get("checkpoint_steps", 25))
        max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
        if logging_steps < 0 or checkpoint_steps < 0:
            raise ValueError("logging_steps and checkpoint_steps must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        save_state_dict_only = bool(
            training_config.get("save_state_dict_only", True)
        )

        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)

            print_main(f"Starting epoch {epoch + 1}/{max_epochs}", rank)
            global_step, train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                protein_dir=protein_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                mixed_precision=mixed_precision,
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                epoch=epoch,
                global_step=global_step,
                rank=rank,
                distributed=distributed,
                writer=writer,
                logging_steps=logging_steps,
                checkpoint_steps=checkpoint_steps,
                checkpoint_dir=checkpoint_dir,
                early_stopper=early_stopper,
                config=resolved_config,
                best_valid_loss=best_valid_loss,
                save_state_dict_only=save_state_dict_only,
            )
            valid_loss = evaluate(
                model,
                valid_loader,
                protein_dir,
                device,
                mixed_precision,
                distributed,
            )

            if is_main_process(rank):
                print(
                    f"epoch={epoch} train_loss={train_loss:.6f} "
                    f"valid_loss={valid_loss:.6f}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/epoch_loss", train_loss, epoch)
                    writer.add_scalar("valid/loss", valid_loss, epoch)
                    writer.add_scalar(
                        "valid/perplexity",
                        math.exp(min(valid_loss, 20.0)),
                        epoch,
                    )

                should_stop = early_stopper.step(valid_loss, epoch=epoch)
                improved = (
                    best_valid_loss is None or valid_loss < best_valid_loss
                )
                if improved:
                    best_valid_loss = valid_loss

                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    early_stopper,
                    epoch=epoch,
                    global_step=global_step,
                    config=resolved_config,
                    best_valid_loss=best_valid_loss,
                    last_train_loss=train_loss,
                    last_valid_loss=valid_loss,
                    save_state_dict_only=save_state_dict_only,
                )
                if improved:
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        early_stopper,
                        epoch=epoch,
                        global_step=global_step,
                        config=resolved_config,
                        best_valid_loss=best_valid_loss,
                        last_train_loss=train_loss,
                        last_valid_loss=valid_loss,
                        save_state_dict_only=save_state_dict_only,
                    )
                    print(
                        f"Updated best checkpoint: valid_loss={valid_loss:.6f}",
                        flush=True,
                    )
            else:
                should_stop = False

            if distributed:
                stop_tensor = torch.tensor(int(should_stop), device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                print_main("Early stopping triggered.", rank)
                break

        return 0
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
