#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

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
from utils.early_stopping import EarlyStopping


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

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


def iter_smiles_offsets(path: Path, has_header: bool) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        if has_header:
            handle.readline()
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def build_or_load_splits(
    data_path: Path,
    split_path: Path,
    has_header: bool,
    seed: int,
    split_ratio: tuple[float, float, float],
    distributed: bool,
    rank: int,
) -> dict[str, torch.Tensor]:
    if is_main_process(rank) and not split_path.exists():
        print(f"Building split index from {data_path}", flush=True)
        offsets = torch.tensor(iter_smiles_offsets(data_path, has_header), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(offsets.numel(), generator=generator)
        offsets = offsets[permutation]

        train_ratio, valid_ratio, _ = split_ratio
        n_total = offsets.numel()
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        splits = {
            "train": offsets[:n_train].clone(),
            "valid": offsets[n_train : n_train + n_valid].clone(),
            "test": offsets[n_train + n_valid :].clone(),
            "metadata": {
                "data_path": str(data_path),
                "has_header": has_header,
                "seed": seed,
                "split_ratio": split_ratio,
                "n_total": n_total,
            },
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(splits, split_path)
        print(
            f"Split index saved to {split_path}: "
            f"train={len(splits['train']):,}, valid={len(splits['valid']):,}, "
            f"test={len(splits['test']):,}",
            flush=True,
        )

    barrier(distributed)
    return torch.load(split_path, map_location="cpu")


class OffsetSmilesDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        offsets: torch.Tensor,
        tokenizer: SmilesTokenizer,
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.offsets = offsets.cpu().long()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._handle = None

    def __len__(self) -> int:
        return int(self.offsets.numel())

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _file(self):
        if self._handle is None:
            self._handle = self.data_path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> list[int]:
        handle = self._file()
        handle.seek(int(self.offsets[index]))
        line = handle.readline().decode("utf-8", errors="ignore").strip()
        smiles = line.split(maxsplit=1)[0]
        token_ids = self.tokenizer.encode(smiles, add_bos=True, add_eos=True)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
            token_ids[-1] = self.tokenizer.eos_token_id
        return token_ids

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()


class SmilesCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[list[int]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_length = max(len(item) for item in batch)
        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for row, token_ids in enumerate(batch):
            length = len(token_ids)
            input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row, :length] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=SmilesCollator(dataset.tokenizer.pad_token_id),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


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


def make_grad_scaler(mixed_precision: str) -> torch.cuda.amp.GradScaler | None:
    if torch.cuda.is_available() and mixed_precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopper: EarlyStopping,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_val_loss: float | None,
    last_train_loss: float | None,
    last_valid_loss: float | None,
    save_state_dict_only: bool = True,
) -> None:
    state = model_state(model)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopping_state_dict": early_stopper.state_dict(),
        "best_val_loss": best_val_loss,
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
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    early_stopper: EarlyStopping | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int, float | None]:
    checkpoint = torch.load(path, map_location=map_location)
    load_model_state(model, checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if early_stopper is not None and "early_stopping_state_dict" in checkpoint:
        early_stopper.load_state_dict(checkpoint["early_stopping_state_dict"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        checkpoint.get("best_val_loss"),
    )


def load_initial_state_dict(
    path: Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    load_model_state(model, state, strict=strict)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    mixed_precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accumulation_steps: int,
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
    best_val_loss: float | None,
    save_state_dict_only: bool,
) -> tuple[int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    epoch_loss = 0.0
    epoch_count = 0
    last_logged_loss = math.nan

    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

        with precision_context(mixed_precision, device):
            outputs = model(**batch)
            loss = outputs.loss / grad_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_for_log = loss.detach() * grad_accumulation_steps
        reduced_loss = reduce_sum(loss_for_log, distributed) / dist.get_world_size() if distributed else loss_for_log
        reduced_loss_value = float(reduced_loss.item())
        running_loss += reduced_loss_value
        running_count += 1
        epoch_loss += reduced_loss_value
        epoch_count += 1

        if step % grad_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main_process(rank) and logging_steps and global_step % logging_steps == 0:
                last_logged_loss = running_loss / max(running_count, 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch} step={global_step} train_loss={last_logged_loss:.6f} lr={lr:.6e}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", last_logged_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                running_loss = 0.0
                running_count = 0

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
                    best_val_loss=best_val_loss,
                    last_train_loss=last_logged_loss if not math.isnan(last_logged_loss) else None,
                    last_valid_loss=None,
                    save_state_dict_only=save_state_dict_only,
                )

    if len(loader) % grad_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if running_count:
        last_logged_loss = running_loss / running_count
    return global_step, epoch_loss / max(epoch_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    distributed: bool,
) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)

    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with precision_context(mixed_precision, device):
            outputs = model(**batch)
        batch_size = batch["input_ids"].size(0)
        total_loss += outputs.loss.detach() * batch_size
        total_count += batch_size

    total_loss = reduce_sum(total_loss, distributed)
    total_count = reduce_sum(total_count, distributed)
    return float((total_loss / total_count).item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain an unconditional GPT-2 SMILES generator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/smiles_gen_pretrain_config.yaml"),
        help="Training config YAML.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a full checkpoint such as latest.pt.",
    )
    parser.add_argument(
        "--init-state-dict",
        type=Path,
        default=None,
        help="Initialize model weights from a model state_dict or checkpoint without optimizer state.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.exists() else project_path(args.config)
    config = load_yaml(config_path)
    distributed, rank, local_rank, world_size = init_distributed()

    seed = int(config.get("seed", 42))
    set_seed(seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = project_path(config["training"].get("output_dir", "outputs/smiles_gen_pretrain"))
    checkpoint_dir = output_dir / "checkpoints"
    split_path = project_path(config["data"].get("split_index_path", output_dir / "splits" / "zinc_split_offsets.pt"))
    tensorboard_dir = output_dir / "tensorboard"

    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    print_main(f"World size: {world_size}, device: {device}", rank)

    data_cfg = config["data"]
    data_path = project_path(data_cfg["train_smi_path"])
    tokenizer = SmilesTokenizer(project_path(data_cfg["vocab_path"]))
    splits = build_or_load_splits(
        data_path=data_path,
        split_path=split_path,
        has_header=bool(data_cfg.get("has_header", True)),
        seed=seed,
        split_ratio=tuple(data_cfg.get("split_ratio", [0.8, 0.1, 0.1])),
        distributed=distributed,
        rank=rank,
    )

    max_length = int(data_cfg.get("max_length", 128))
    train_dataset = OffsetSmilesDataset(data_path, splits["train"], tokenizer, max_length)
    valid_dataset = OffsetSmilesDataset(data_path, splits["valid"], tokenizer, max_length)

    train_cfg = config["training"]
    train_loader, train_sampler = make_dataloader(
        train_dataset,
        batch_size=int(train_cfg.get("per_device_train_batch_size", train_cfg.get("batch_size", 128))),
        num_workers=int(train_cfg.get("num_workers", 4)),
        distributed=distributed,
        shuffle=True,
        drop_last=True,
    )
    valid_loader, valid_sampler = make_dataloader(
        valid_dataset,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 256)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        distributed=distributed,
        shuffle=False,
        drop_last=False,
    )

    model_cfg = config["model"]
    model = SmilesGPT2Generator.from_tokenizer(
        tokenizer=tokenizer,
        max_position_embeddings=int(model_cfg.get("max_position_embeddings", 128)),
        n_embd=int(model_cfg.get("n_embd", 512)),
        n_layer=int(model_cfg.get("n_layer", 8)),
        n_head=int(model_cfg.get("n_head", 8)),
        resid_pdrop=float(model_cfg.get("resid_pdrop", 0.1)),
        embd_pdrop=float(model_cfg.get("embd_pdrop", 0.1)),
        attn_pdrop=float(model_cfg.get("attn_pdrop", 0.1)),
        layer_norm_epsilon=float(model_cfg.get("layer_norm_epsilon", 1e-5)),
        add_cross_attention=bool(model_cfg.get("add_cross_attention", False)),
    ).to(device)

    if args.init_state_dict is not None:
        init_state_path = args.init_state_dict if args.init_state_dict.exists() else project_path(args.init_state_dict)
        load_initial_state_dict(init_state_path, model, map_location=device, strict=True)
        print_main(f"Initialized model state from {init_state_path}", rank)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_epochs = int(train_cfg.get("max_epochs", 20))
    grad_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    updates_per_epoch = math.ceil(len(train_loader) / grad_accumulation_steps)
    total_steps = max_epochs * updates_per_epoch
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_cfg = config.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=int(early_cfg.get("patience", 7)),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
        mode="min",
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = None
    if args.resume is not None:
        resume_path = args.resume if args.resume.exists() else project_path(args.resume)
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            map_location=device,
        )
        print_main(f"Resumed from {resume_path} at epoch={start_epoch}, step={global_step}", rank)

    mixed_precision = str(train_cfg.get("mixed_precision", "bf16")).lower()
    if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        mixed_precision = "fp16"
        print_main("bf16 is not supported on this GPU; falling back to fp16.", rank)
    scaler = make_grad_scaler(mixed_precision)

    writer = SummaryWriter(str(tensorboard_dir)) if is_main_process(rank) else None
    logging_steps = int(train_cfg.get("logging_steps", 100))
    checkpoint_steps = int(train_cfg.get("checkpoint_steps", 1000))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_state_dict_only = bool(train_cfg.get("save_state_dict_only", True))

    if is_main_process(rank) and args.resume is None:
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            early_stopper,
            epoch=-1,
            global_step=global_step,
            config=config,
            best_val_loss=best_val_loss,
            last_train_loss=None,
            last_valid_loss=None,
            save_state_dict_only=save_state_dict_only,
        )

    try:
        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)

            print_main(f"Starting epoch {epoch + 1}/{max_epochs}", rank)
            global_step, train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                mixed_precision=mixed_precision,
                scaler=scaler,
                grad_accumulation_steps=grad_accumulation_steps,
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
                config=config,
                best_val_loss=best_val_loss,
                save_state_dict_only=save_state_dict_only,
            )

            valid_loss = evaluate(model, valid_loader, device, mixed_precision, distributed)
            if is_main_process(rank):
                print(
                    f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/epoch_loss", train_loss, epoch)
                    writer.add_scalar("valid/loss", valid_loss, epoch)
                    writer.add_scalar("valid/perplexity", math.exp(min(valid_loss, 20)), epoch)

                should_stop = early_stopper.step(valid_loss, epoch=epoch)
                improved = best_val_loss is None or valid_loss < best_val_loss
                if improved:
                    best_val_loss = valid_loss

                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    early_stopper,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    best_val_loss=best_val_loss,
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
                        config=config,
                        best_val_loss=best_val_loss,
                        last_train_loss=train_loss,
                        last_valid_loss=valid_loss,
                        save_state_dict_only=save_state_dict_only,
                    )
                    print(f"Updated best checkpoint: valid_loss={valid_loss:.6f}", flush=True)
            else:
                should_stop = False

            if distributed:
                stop_tensor = torch.tensor(int(should_stop), device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                print_main("Early stopping triggered.", rank)
                break

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
