#!/usr/bin/env python3
"""Second-stage sequence preference tuning with first-stage MTD replay.

L = MTD_TF + spri_tf_loss_weight * SPRi_PP_TF + lambda(step) * preference.
Sequence scores include EOS, exclude BOS/padding, and average over real tokens.
Only PP and MTD are positive generation targets. Negative conditions participate
only in softplus ranking. Each mixed micro-batch uses one model forward/backward.
Run --check-data to validate data/dependencies without allocating the full model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.smiles_gen import SmilesTokenizer
from model.smiles_gen_finetune import (
    PositiveRecord, PositiveTargetCollator, atomic_torch_save, barrier,
    build_joint_model, build_optimizer, cleanup_distributed,
    filter_records_with_available_proteins, fingerprint_records, init_distributed,
    load_unique_protein_encodings, load_yaml, log_learning_rates, make_grad_scaler,
    precision_context, print_main, project_path, read_positive_records, set_seed,
    validate_protein_files, validate_split_ratio,
)
from utils.early_stopping import EarlyStopping


@dataclass(frozen=True)
class ContrastiveRecord:
    source_row_number: int
    smiles: str
    pp: tuple[str, str]
    pn: tuple[str, str]
    nn: tuple[str, str]


def load_local(path: Path):
    # These are user-selected, trusted local PyTorch model/split files.
    return torch.load(path, map_location="cpu", weights_only=False)


def read_contrastive_records(path: Path) -> list[ContrastiveRecord]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["smiles", "pp", "pn", "nn"]:
            raise ValueError("Contrastive CSV columns must be smiles,pp,pn,nn")
        for line, row in enumerate(reader, 2):
            pairs = []
            for key in ("pp", "pn", "nn"):
                values = tuple(sorted(v.strip() for v in (row[key] or "").split(",")))
                if len(values) != 2 or len(set(values)) != 2 or any(
                    not v or Path(v).name != v for v in values
                ):
                    raise ValueError(f"Row {line}: {key} must contain two distinct target IDs")
                pairs.append(values)
            pp, pn, nn = map(set, pairs)
            if len(pp & pn) != 1 or pp & nn:
                raise ValueError(f"Row {line}: inconsistent PP/PN/NN relationships")
            smiles = (row["smiles"] or "").strip()
            if not smiles:
                raise ValueError(f"Row {line}: empty SMILES")
            records.append(ContrastiveRecord(line, smiles, *pairs))
    if not records:
        raise ValueError("Empty contrastive CSV")
    # Check the inferred positive/negative sets across ALL rows of each molecule.
    labels = defaultdict(lambda: [set(), set()])
    seen = set()
    for r in records:
        key = (r.smiles, r.pp, r.pn, r.nn)
        if key in seen:
            raise ValueError(f"Duplicate contrastive record at row {r.source_row_number}")
        seen.add(key)
        labels[r.smiles][0].update(r.pp)
        labels[r.smiles][1].update(set(r.pn) - set(r.pp))
        labels[r.smiles][1].update(r.nn)
    if any(p & n for p, n in labels.values()):
        raise ValueError("Conflicting positive/negative labels for the same molecule")
    return records


def prepare_sequences(records, tokenizer, max_length, policy, task, audit, cache):
    """Drop whole SMILES groups on invalid chemistry, unknown tokens or length."""
    kept = []
    for record in records:
        smiles = record.smiles
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            tokens = tokenizer.tokenize(smiles)
            unknown = sorted(set(tokens) - set(tokenizer.token_to_id))
            reason = []
            if mol is None or mol.GetNumAtoms() == 0:
                reason.append("invalid SMILES")
            if unknown:
                reason.append(f"out-of-vocabulary tokens: {unknown}")
            if len(tokens) + 2 > max_length:
                reason.append(f"length including BOS/EOS={len(tokens)+2} > {max_length}")
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
            cache[smiles] = (canonical, tokenizer.encode(smiles), "; ".join(reason))
        canonical, token_ids, reason = cache[smiles]
        if reason:
            if policy == "error":
                raise ValueError(f"{task} row {record.source_row_number}: {smiles}: {reason}")
            audit.append((task, record.source_row_number, smiles, reason))
        else:
            kept.append(record)
    return kept


def grouped_splits(records, cache, ratios, seed, method):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        canonical = cache[r.smiles][0]
        if method == "scaffold":
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(canonical)
            # Acyclic compounds have an empty Murcko scaffold; group by molecule.
            key = scaffold or canonical
        else:
            key = canonical
        groups[key].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    result = {k: [] for k in ("train", "valid", "test")}
    if method == "molecule":
        ntrain, nvalid = int(len(keys)*ratios[0]), int(len(keys)*ratios[1])
        if ratios[2] == 0:
            nvalid = len(keys) - ntrain
        for part, selected in (("train", keys[:ntrain]), ("valid", keys[ntrain:ntrain+nvalid]),
                               ("test", keys[ntrain+nvalid:])):
            result[part] = [i for key in selected for i in groups[key]]
    else:
        targets = [int(len(records)*ratios[0]), int(len(records)*ratios[1])]
        active = [part for part, ratio in zip(result, ratios) if ratio > 0]
        for position, key in enumerate(keys):
            empty = [part for part in active if not result[part]]
            if len(keys)-position <= len(empty):
                split = empty[0]
            else:
                split = "train" if len(result["train"]) < targets[0] else (
                    "valid" if ratios[2] == 0 or len(result["valid"]) < targets[1] else "test"
                )
            result[split].extend(groups[key])
    if not result["train"] or not result["valid"] or (ratios[2] > 0 and not result["test"]):
        raise ValueError("Too few molecule/scaffold groups for nonempty requested splits")
    return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def read_mtd_splits(path, records):
    splits = load_local(path)
    meta = splits.get("metadata", {})
    # Absolute cluster paths may differ; the ordered record fingerprint must not.
    if meta.get("record_count") != len(records) or meta.get("record_fingerprint") != fingerprint_records(records):
        raise ValueError(
            "MTD CSV / available protein files do not reproduce the first-stage "
            "ordered records. Refusing to reuse misaligned split indices. "
            "Use the same MTD CSV and protein encodings as first-stage training."
        )
    indices = torch.cat([splits[k].long() for k in ("train", "valid", "test")])
    if len(indices) != len(records) or not torch.equal(indices.sort().values, torch.arange(len(records))):
        raise ValueError("First-stage MTD split indices must partition all records")
    return splits


class MixedDataset(Dataset):
    def __init__(self, contrastive, mtd, cache):
        self.contrastive, self.mtd, self.cache = contrastive, mtd, cache

    def __len__(self):
        return len(self.contrastive) + len(self.mtd)

    def __getitem__(self, i):
        if i < len(self.contrastive):
            record = self.contrastive[i]
            return "spri", self.cache[record.smiles][1], (record.pp, record.pn, record.nn)
        record = self.mtd[i - len(self.contrastive)]
        return "mtd", self.cache[record.smiles][1], (record.target_ids,)


class MixedCollator:
    def __init__(self, pad_token_id):
        self.collator = PositiveTargetCollator(pad_token_id)

    def __call__(self, rows):
        spri = [r for r in rows if r[0] == "spri"]
        mtd = [r for r in rows if r[0] == "mtd"]
        sequences = [(r[1], r[2][branch]) for branch in range(3) for r in spri]
        sequences += [(r[1], r[2][0]) for r in mtd]
        result = self.collator(sequences)
        result.update(contrastive_count=len(spri), mtd_count=len(mtd))
        return result


class MixedBatchSampler(Sampler[list[int]]):
    """One SPRi pass/epoch, padding last global batch; reshuffle MTD each epoch.

    Batch counts stay exact on every device, including the final batch. Padding
    repeats at most world_size * contrastive_batch_size - 1 SPRi rows per epoch.
    Replay draws without replacement until its pool is exhausted, then reshuffles.
    Deterministic epoch streams make mid-epoch resume possible without skipping RNG.
    """
    def __init__(self, contrastive, mtd, offset, csize, msize, seed, rank=0, world=1):
        self.contrastive, self.mtd = list(contrastive), list(mtd)
        self.offset, self.csize, self.msize = offset, csize, msize
        self.seed, self.rank, self.world = seed, rank, world
        self.epoch = self.start_batch = 0
        if not self.contrastive or not self.mtd:
            raise ValueError("Both training pools must be nonempty")

    @property
    def full_length(self):
        return math.ceil(len(self.contrastive) / (self.csize * self.world))

    def __len__(self):
        return self.full_length - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        total = self.full_length * self.world * self.csize
        c = np.resize(rng.permutation(self.contrastive), total)
        required = self.full_length * self.world * self.msize
        replay = []
        while len(replay) < required:
            replay.extend(rng.permutation(self.mtd).tolist())
        m = np.asarray(replay[:required])
        for step in range(self.start_batch, self.full_length):
            ci = (step * self.world + self.rank) * self.csize
            mi = (step * self.world + self.rank) * self.msize
            yield c[ci:ci+self.csize].tolist() + (m[mi:mi+self.msize] + self.offset).tolist()


def sequence_scores(logits, labels):
    """Causal shift: position 0 predicts first SMILES token, last predicts EOS."""
    targets = labels[:, 1:]
    mask = targets != -100
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Sequence has no scored tokens")
    nll = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets,
        reduction="none", ignore_index=-100,
    )
    return -(nll * mask).sum(dim=1) / counts


def preference_loss(pp, pn, nn, beta=1.0, margin=0.0, enforce_pn_gt_nn=False):
    terms = [F.softplus(beta * (margin - (pp-pn))),
             F.softplus(beta * (margin - (pp-nn)))]
    if enforce_pn_gt_nn:
        terms.append(F.softplus(beta * (margin - (pn-nn))))
    return torch.stack(terms).mean()


def compute_losses(model, batch, protein_dir, device, tc):
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=None,
        unique_protein_encodings=load_unique_protein_encodings(batch["unique_target_ids"], protein_dir),
        sample_target_indices=batch["sample_target_indices"],
    ).logits
    scores = sequence_scores(logits, batch["labels"].to(device))
    c, m = batch["contrastive_count"], batch["mtd_count"]
    zero = scores.sum() * 0
    values = {"mtd_tf_loss": -scores[3*c:].mean() if m else zero}
    if c:
        pp, pn, nn = scores[:3*c].split(c)
        values.update(
            spri_tf_loss=-pp.mean(),
            contrastive_loss=preference_loss(pp, pn, nn, tc["contrastive_beta"],
                                            tc["contrastive_margin"], tc["enforce_pn_gt_nn"]),
            pp_score=pp.mean(), pn_score=pn.mean(), nn_score=nn.mean(),
            pp_pn_accuracy=(pp > pn).float().mean(),
            pp_nn_accuracy=(pp > nn).float().mean(),
            pn_nn_accuracy=(pn > nn).float().mean(),
            pp_pn_gap=(pp-pn).mean(), pp_nn_gap=(pp-nn).mean(),
        )
    return values


def contrastive_weight(tc, global_step, total_steps):
    ramp = math.ceil(total_steps * tc["contrastive_warmup_ratio"])
    return tc["contrastive_loss_weight"] * (min(global_step / ramp, 1.0) if ramp else 1.0)


def total_loss(values, tc, weight):
    return (values["mtd_tf_loss"] + tc["spri_tf_loss_weight"] * values["spri_tf_loss"]
            + weight * values["contrastive_loss"])


def average_metrics(totals, device, distributed):
    keys = sorted(totals)
    tensor = torch.tensor([totals[k] for k in keys], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor)
    return {k: float(v[0] / v[1]) for k, v in zip(keys, tensor) if v[1] > 0}


def add_metrics(totals, values, c, m):
    for key, value in values.items():
        count = m if key == "mtd_tf_loss" else c
        totals[key][0] += float(value.detach()) * count
        totals[key][1] += count


@torch.no_grad()
def evaluate(model, dataset, splits, tc, protein_dir, device, rank, world):
    # Unwrap DDP: validation shards may have unequal batch counts.
    model = model.module if isinstance(model, DDP) else model
    model.eval()
    totals = defaultdict(lambda: [0.0, 0.0])
    for name, offset in (("spri", 0), ("mtd", len(dataset.contrastive))):
        indices = [int(i) + offset for i in splits[name]["valid"][rank::world]]
        loader = DataLoader(dataset, batch_size=tc["per_device_eval_batch_size"],
                            sampler=indices, collate_fn=MixedCollator(tc["pad_token_id"]),
                            num_workers=tc["num_workers"])
        for batch in loader:
            with precision_context(tc["mixed_precision"], device):
                values = compute_losses(model, batch, protein_dir, device, tc)
            add_metrics(totals, values, batch["contrastive_count"], batch["mtd_count"])
    # All ranks reduce the same key set, including ranks with empty tiny shards.
    for k in METRIC_NAMES:
        totals[k]
    metrics = average_metrics(totals, device, world > 1)
    # Fixed FULL weight across epochs: warmup must not change early-stop objective.
    metrics["loss"] = (metrics["mtd_tf_loss"] + tc["spri_tf_loss_weight"] * metrics["spri_tf_loss"]
                       + tc["contrastive_loss_weight"] * metrics["contrastive_loss"])
    return metrics


METRIC_NAMES = ("mtd_tf_loss", "spri_tf_loss", "contrastive_loss", "pp_score", "pn_score",
                "nn_score", "pp_pn_accuracy", "pp_nn_accuracy", "pn_nn_accuracy", "pp_pn_gap", "pp_nn_gap")


def save_training_checkpoint(path, model, optimizer, scheduler, scaler, stopper,
                             epoch, next_batch, step, config, train_loss, valid_loss, rank, world):
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank != 0:
        return
    base = model.module if isinstance(model, DDP) else model
    state = base.state_dict()
    payload = dict(model_state_dict=state, optimizer_state_dict=optimizer.state_dict(),
                   scheduler_state_dict=scheduler.state_dict(), early_stopping_state_dict=stopper.state_dict(),
                   scaler_state_dict=scaler.state_dict() if scaler else None,
                   epoch=epoch, next_batch=next_batch, global_step=step,
                   best_valid_loss=stopper.best_score, last_train_loss=train_loss,
                   last_valid_loss=valid_loss, config=config, rng_states=states, world_size=world)
    atomic_torch_save(payload, path)
    if config["training"]["save_state_dict_only"]:
        atomic_torch_save(state, path.with_name(path.stem + "_state_dict.pt"))


def validate_training_config(tc):
    for key in ("contrastive_batch_size", "mtd_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "max_epochs"):
        if type(tc[key]) is not int or tc[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("num_workers", "logging_steps", "checkpoint_steps"):
        if type(tc[key]) is not int or tc[key] < 0:
            raise ValueError(f"training.{key} must be a non-negative integer")
    for key in ("spri_tf_loss_weight", "contrastive_loss_weight", "contrastive_margin"):
        if not math.isfinite(float(tc[key])) or tc[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    for key in ("contrastive_beta", "max_grad_norm", "gpt_learning_rate",
                "cross_attention_learning_rate", "target_encoder_learning_rate"):
        if not math.isfinite(float(tc[key])) or tc[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key in ("warmup_ratio", "contrastive_warmup_ratio"):
        if not 0 <= tc[key] <= 1:
            raise ValueError(f"training.{key} must be in [0,1]")
    if type(tc["enforce_pn_gt_nn"]) is not bool:
        raise ValueError("enforce_pn_gt_nn must be a YAML boolean")
    if tc["mixed_precision"] not in {"no", "none", "fp32", "fp16", "bf16"}:
        raise ValueError("Invalid mixed_precision")


def prepare_data(config, first, tokenizer, rank):
    dc = config["data"]
    protein_dir = project_path(dc["protein_dir"])
    cpath, mpath = project_path(dc["contrastive_csv_path"]), project_path(dc["mtd_csv_path"])
    cache, audit = {}, []
    policy = dc["invalid_smiles_policy"]
    if policy not in {"skip", "error"} or dc["split_method"] not in {"molecule", "scaffold"}:
        raise ValueError("Invalid data policy/split method")
    spri = prepare_sequences(read_contrastive_records(cpath), tokenizer, dc["max_length"], policy, "spri", audit, cache)
    if not spri:
        raise ValueError("No usable SPRi records")
    # Re-check label consistency after canonical grouping.
    canonical_labels = defaultdict(lambda: [set(), set()])
    for r in spri:
        p, n = canonical_labels[cache[r.smiles][0]]
        p.update(r.pp); n.update(set(r.pn)-set(r.pp)); n.update(r.nn)
    if any(p & n for p, n in canonical_labels.values()):
        raise ValueError("Conflicting labels after canonical structure grouping")
    csplits = grouped_splits(spri, cache, validate_split_ratio(dc["split_ratio"]), config["seed"], dc["split_method"])
    original_mtd = read_positive_records(mpath)
    manifest_path = dc.get("mtd_protein_manifest_path")
    if manifest_path:
        available = {Path(line.strip()).stem for line in project_path(manifest_path).read_text().splitlines() if line.strip()}
        mtd = [r for r in original_mtd if all(t in available for t in r.target_ids)]
        missing_count = len(original_mtd) - len(mtd)
    else:
        mtd, missing = filter_records_with_available_proteins(original_mtd, protein_dir)
        missing_count = len(missing)
    del original_mtd
    msplits_original = read_mtd_splits(project_path(dc["mtd_split_index_path"]), mtd)
    # Keep MTD ordering for original split indices; mark unusable rows rather than renumber.
    usable = prepare_sequences(mtd, tokenizer, dc["max_length"], policy, "mtd", audit, cache)
    good = {r.source_row_number for r in usable}
    holdout = {cache[spri[int(i)].smiles][0] for part in ("valid", "test") for i in csplits[part]}
    if dc["split_method"] == "scaffold":
        holdout_scaffolds = {MurckoScaffold.MurckoScaffoldSmilesFromSmiles(x) or x for x in holdout}
    excluded = []
    msplits = {}
    for part in ("train", "valid", "test"):
        selected = []
        for i in msplits_original[part].tolist():
            r = mtd[i]
            if r.source_row_number not in good:
                continue
            canonical = cache[r.smiles][0]
            group = (MurckoScaffold.MurckoScaffoldSmilesFromSmiles(canonical) or canonical) if dc["split_method"] == "scaffold" else canonical
            overlaps = group in (holdout_scaffolds if dc["split_method"] == "scaffold" else holdout)
            if part == "train" and dc["exclude_spri_holdout_from_mtd"] and overlaps:
                excluded.append(i)
            else:
                selected.append(i)
        msplits[part] = torch.tensor(selected, dtype=torch.long)
    if not len(msplits["train"]) or not len(msplits["valid"]):
        raise ValueError("MTD train/valid pool is empty after filtering")
    proteins = [PositiveRecord(r.source_row_number, r.smiles, tuple(set(r.pp+r.pn+r.nn))) for r in spri]
    proteins += usable
    splits = {"spri": csplits, "mtd": msplits}
    metadata = {"spri_csv_sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(),
                "mtd_record_fingerprint": fingerprint_records(mtd), "seed": config["seed"],
                "split_method": dc["split_method"], "split_ratio": dc["split_ratio"],
                "vocab_sha256": hashlib.sha256(tokenizer.vocab_path.read_bytes()).hexdigest(),
                "max_length": dc["max_length"], "excluded_mtd_replay_rows": len(excluded),
                "exclude_spri_holdout_from_mtd": dc["exclude_spri_holdout_from_mtd"],
                "skipped_missing_mtd_protein_rows": missing_count,
                "spri_molecules": len({cache[r.smiles][0] for r in spri}),
                "skipped_sequence_rows": len(audit)}
    print_main(f"Data: {metadata}; splits=" + str({task: {k: len(v) for k, v in parts.items()} for task, parts in splits.items()}), rank)
    missing_ids = sorted({t for r in proteins for t in r.target_ids if not (protein_dir / f"{t}.npy").is_file()})
    if missing_ids:
        spri_missing = sorted({t for r in spri for t in r.pp+r.pn+r.nn if t in set(missing_ids)})
        raise FileNotFoundError(
            f"Missing {len(missing_ids)} protein encodings in {protein_dir}. "
            f"SPRi missing: {', '.join(spri_missing) or 'none'}. "
            f"First 20 across both tasks: {', '.join(missing_ids[:20])}. "
            "Supply the complete encodings in data.protein_dir before training."
        )
    validate_protein_files(proteins, protein_dir, int(first["target_encoder"]["input_dim"]))
    payload = {**splits, "metadata": metadata,
               "spri_source_row_numbers": torch.tensor([r.source_row_number for r in spri]),
               "mtd_source_row_numbers": torch.tensor([r.source_row_number for r in mtd])}
    return MixedDataset(spri, mtd, cache), splits, metadata, (payload, audit, excluded)


def save_data_audit(output, payload, audit, excluded):
    directory = output / "splits"
    path = directory / "secondary_split_indices.pt"
    if path.exists():
        old = load_local(path)
        if old["metadata"] != payload["metadata"] or any(
            not torch.equal(old[task][part], payload[task][part])
            for task in ("spri", "mtd") for part in ("train", "valid", "test")
        ):
            raise ValueError("Output split file differs; use a fresh training.output_dir")
    else:
        atomic_torch_save(payload, path)
    with (directory / "skipped_smiles.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "source_row_number", "smiles", "reason"])
        writer.writerows(audit)
    atomic_torch_save(torch.tensor(excluded, dtype=torch.long), directory / "excluded_mtd_replay_indices.pt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/smiles_gen_contrastive_secondary_finetune_config.yaml"))
    parser.add_argument("--resume", type=Path, help="Resume a full SECONDARY latest.pt or best.pt")
    parser.add_argument("--check-data", action="store_true", help="Read-only data/config/protein checks; do not load model weights")
    args = parser.parse_args(argv)
    config = load_yaml(project_path(args.config))
    tc, dc = config["training"], config["data"]
    validate_training_config(tc)
    distributed, rank, local_rank, world = init_distributed()
    writer = None
    try:
        set_seed(config["seed"] + rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        section = config["pretrained"]
        source = project_path(section["output_dir"])
        output = project_path(tc["output_dir"])
        if output.resolve() == source.resolve():
            raise ValueError("Secondary output_dir must differ from first-stage model directory")
        first = load_yaml(source / section["resolved_config"])
        model_config = copy.deepcopy(first["resolved_pretrained"]["model"])
        # Legacy first-stage config has no count: it used cross-attention in ALL layers.
        count = first.get("model", {}).get("num_cross_attention_layers", model_config.get("num_cross_attention_layers", model_config["n_layer"]))
        if count == 0:
            raise ValueError("Sequence condition ranking requires protein cross-attention")
        tokenizer = SmilesTokenizer(project_path(dc["vocab_path"]))
        vocab_source = Path(first["resolved_pretrained"]["vocab_path"])
        # Cluster paths are not portable; the first-stage pretraining config is a local fallback.
        if not vocab_source.is_file():
            preconfig = load_yaml(project_path(first["pretrained"]["output_dir"]) / first["pretrained"].get("resolved_config", "resolved_config.yaml"))
            vocab_source = project_path(preconfig["data"]["vocab_path"])
        if tokenizer.token_to_id != SmilesTokenizer(vocab_source).token_to_id:
            raise ValueError("Vocabulary token IDs must match the first-stage model exactly")
        if not 2 <= dc["max_length"] <= model_config["max_position_embeddings"]:
            raise ValueError("max_length exceeds checkpoint positional capacity or is too small")
        tc["pad_token_id"] = tokenizer.pad_token_id
        dataset, splits, metadata, audit_bundle = prepare_data(config, first, tokenizer, rank)
        if args.check_data:
            return 0
        if tc["mixed_precision"] == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            tc["mixed_precision"] = "fp16"
            print_main("bf16 unsupported; using fp16 with gradient scaling.", rank)
        checkpoint_path = project_path(args.resume) if args.resume else source / section["checkpoint"]
        checkpoint = load_local(checkpoint_path)
        if args.resume and ("rng_states" not in checkpoint or checkpoint.get("world_size") != world):
            raise ValueError("Resume requires a secondary full checkpoint and the same world size")
        if args.resume and checkpoint["config"]["resolved_secondary_data"] != metadata:
            raise ValueError("Resume data differs from checkpoint")
        if args.resume and checkpoint["config"]["training"] != tc:
            raise ValueError("Resume training settings differ from checkpoint")
        if rank == 0:
            if not args.resume and (output / "checkpoints/latest.pt").exists():
                raise ValueError("Output already has checkpoints; use --resume or a fresh output_dir")
            save_data_audit(output, *audit_bundle)
        del audit_bundle
        barrier(distributed)
        model = build_joint_model(tokenizer, {"model": model_config}, first["target_encoder"], count)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        model.smiles_model.gpt2.config.use_cache = False
        model.to(device)
        optimizer = build_optimizer(model, tc)
        sampler = MixedBatchSampler(splits["spri"]["train"].tolist(), splits["mtd"]["train"].tolist(),
                                    len(dataset.contrastive), tc["contrastive_batch_size"], tc["mtd_batch_size"],
                                    config["seed"], rank, world)
        updates_per_epoch = math.ceil(sampler.full_length / tc["gradient_accumulation_steps"])
        total_steps = updates_per_epoch * tc["max_epochs"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*tc["warmup_ratio"]), total_steps)
        scaler = make_grad_scaler(tc["mixed_precision"])
        stopper = EarlyStopping(**config["early_stopping"], mode="min")
        start_epoch = start_batch = global_step = 0
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            stopper.load_state_dict(checkpoint["early_stopping_state_dict"])
            if scaler and checkpoint["scaler_state_dict"]:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch, start_batch, global_step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"]
        resolved = copy.deepcopy(config)
        resolved["target_encoder"] = copy.deepcopy(first["target_encoder"])
        resolved["model"] = {"num_cross_attention_layers": count}
        resolved["resolved_pretrained"] = copy.deepcopy(first["resolved_pretrained"])
        resolved["resolved_pretrained"]["model"] = {**model_config, "num_cross_attention_layers": count}
        resolved["resolved_pretrained"]["vocab_path"] = str(tokenizer.vocab_path)
        resolved["resolved_secondary_data"] = metadata
        resolved["resolved_first_stage"] = {"config_path": str(source / section["resolved_config"]), "checkpoint_path": str(source / section["checkpoint"])}
        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if args.resume:
            rng = checkpoint["rng_states"][rank]
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        del checkpoint
        if rank == 0:
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)
            with (output / "resolved_config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(resolved, f, sort_keys=False)
            writer = SummaryWriter(str(output / "tensorboard"), purge_step=global_step if args.resume else None)
        protein_dir = project_path(dc["protein_dir"])
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=MixedCollator(tokenizer.pad_token_id),
                            num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(config["seed"]+rank))
        checkpoint_dir = output / "checkpoints"
        for epoch in range(start_epoch, tc["max_epochs"]):
            if stopper.should_stop:
                break
            sampler.epoch, sampler.start_batch = epoch, start_batch if epoch == start_epoch else 0
            if sampler.start_batch >= sampler.full_length:
                raise ValueError("Invalid resume batch offset")
            model.train(); optimizer.zero_grad(set_to_none=True)
            epoch_totals = defaultdict(lambda: [0.0, 0.0])
            running = defaultdict(lambda: [0.0, 0.0])
            accumulation = tc["gradient_accumulation_steps"]
            for batch_index, batch in enumerate(loader, sampler.start_batch):
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, sampler.full_length-group_start)
                weight = contrastive_weight(tc, global_step, total_steps)
                with precision_context(tc["mixed_precision"], device):
                    values = compute_losses(model, batch, protein_dir, device, tc)
                    loss = total_loss(values, tc, weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, batch={batch_index}")
                (scaler.scale(loss/group_size) if scaler else loss/group_size).backward()
                for totals in (epoch_totals, running):
                    add_metrics(totals, values, batch["contrastive_count"], batch["mtd_count"])
                    totals["loss"][0] += float(loss.detach()); totals["loss"][1] += 1
                update = (batch_index+1) % accumulation == 0 or batch_index+1 == sampler.full_length
                if not update:
                    continue
                if scaler:
                    scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                old_scale = scaler.get_scale() if scaler else None
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                stepped = scaler is None or scaler.get_scale() >= old_scale
                if stepped:
                    scheduler.step(); global_step += 1
                optimizer.zero_grad(set_to_none=True)
                if tc["logging_steps"] and global_step % tc["logging_steps"] == 0:
                    metrics = average_metrics(running, device, distributed)
                    if writer:
                        for key, value in metrics.items(): writer.add_scalar("train/"+key, value, global_step)
                        writer.add_scalar("train/gradient_norm", float(norm), global_step)
                        writer.add_scalar("train/contrastive_loss_weight", weight, global_step)
                        writer.add_scalar("train/perplexity", math.exp(min(metrics["mtd_tf_loss"],20)), global_step)
                        log_learning_rates(writer, optimizer, global_step)
                    print_main(f"epoch={epoch+1} step={global_step} loss={metrics['loss']:.6f} PP>PN={metrics['pp_pn_accuracy']:.3f} PP>NN={metrics['pp_nn_accuracy']:.3f}", rank)
                    running.clear()
                if tc["checkpoint_steps"] and global_step % tc["checkpoint_steps"] == 0 and batch_index+1 < sampler.full_length:
                    save_training_checkpoint(checkpoint_dir/"latest.pt", model, optimizer, scheduler, scaler, stopper,
                                             epoch, batch_index+1, global_step, resolved, float(loss.detach()), None, rank, world)
            start_batch = 0
            train_metrics = average_metrics(epoch_totals, device, distributed)
            valid = evaluate(model, dataset, splits, tc, protein_dir, device, rank, world)
            improved = stopper.is_improvement(valid["loss"])
            stopper.step(valid["loss"], epoch)
            print_main(f"epoch={epoch+1} train_loss={train_metrics['loss']:.6f} valid_loss={valid['loss']:.6f}", rank)
            if writer:
                writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch)
                for key,value in valid.items(): writer.add_scalar("valid/"+key, value, epoch)
                writer.add_scalar("valid/perplexity", math.exp(min(valid["mtd_tf_loss"],20)), epoch)
                writer.flush()
            for name in (["latest", "best"] if improved else ["latest"]):
                save_training_checkpoint(checkpoint_dir/(name+".pt"), model, optimizer, scheduler, scaler, stopper,
                                         epoch+1, 0, global_step, resolved, train_metrics["loss"], valid["loss"], rank, world)
            if stopper.should_stop:
                print_main("Early stopping triggered.", rank)
                break
        return 0
    finally:
        if writer: writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
