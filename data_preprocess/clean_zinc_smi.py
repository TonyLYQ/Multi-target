#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors


ALLOWED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"}
MIN_MW = 100.0
MAX_MW = 650.0
MIN_FORMAL_CHARGE = -2
MAX_FORMAL_CHARGE = 2


def iter_smiles(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            first_field = line.split(maxsplit=1)[0]
            if line_number == 1 and first_field.lower() in {"smiles", "smile"}:
                continue
            yield first_field


def organic_fragments(mol: Chem.Mol) -> list[Chem.Mol]:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return [fragment for fragment in fragments if any(atom.GetSymbol() == "C" for atom in fragment.GetAtoms())]


def largest_organic_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    fragments = organic_fragments(mol)
    if not fragments:
        return None
    return max(
        fragments,
        key=lambda fragment: (
            fragment.GetNumHeavyAtoms(),
            Descriptors.ExactMolWt(fragment),
        ),
    )


def has_allowed_elements(mol: Chem.Mol) -> bool:
    return all(atom.GetSymbol() in ALLOWED_ELEMENTS for atom in mol.GetAtoms())


def has_isotope_atoms(mol: Chem.Mol) -> bool:
    return any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())


def formal_charge(mol: Chem.Mol) -> int:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def print_step(name: str, before: int, after: int) -> None:
    print(f"{name}: filtered={before - after:,}, remaining={after:,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean zinc_ori.smi and save canonical SMILES."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ZINC_data/zinc_ori.smi"),
        help="Input .smi file. Default: ZINC_data/zinc_ori.smi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .smi file. Default: <input_dir>/zinc_filtered.smi",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write a header line to the output file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress every N input molecules. Use 0 to disable.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name("zinc_filtered.smi")

    RDLogger.DisableLog("rdApp.*")

    counts = {
        "input": 0,
        "rdkit": 0,
        "fragment": 0,
        "isotope": 0,
        "elements": 0,
        "charge": 0,
        "mw": 0,
    }
    canonical_smiles: set[str] = set()

    for smiles in iter_smiles(input_path):
        counts["input"] += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        counts["rdkit"] += 1

        fragment = largest_organic_fragment(mol)
        if fragment is None:
            continue
        counts["fragment"] += 1

        if has_isotope_atoms(fragment):
            continue
        counts["isotope"] += 1

        if not has_allowed_elements(fragment):
            continue
        counts["elements"] += 1

        if not (MIN_FORMAL_CHARGE <= formal_charge(fragment) <= MAX_FORMAL_CHARGE):
            continue
        counts["charge"] += 1

        if not (MIN_MW <= Descriptors.MolWt(fragment) <= MAX_MW):
            continue
        counts["mw"] += 1

        canonical_smiles.add(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True))

        if args.progress_every and counts["input"] % args.progress_every == 0:
            print(
                f"Progress: input={counts['input']:,}, "
                f"passed_mw={counts['mw']:,}, unique={len(canonical_smiles):,}",
                flush=True,
            )

    print(f"Input molecules: {counts['input']:,}")
    print_step("RDKit MolFromSmiles", counts["input"], counts["rdkit"])
    print_step(
        "Remove salts/mixtures and keep largest organic fragment",
        counts["rdkit"],
        counts["fragment"],
    )
    print_step("Isotope atom filter", counts["fragment"], counts["isotope"])
    print_step("Element filter (C,N,O,S,P,F,Cl,Br,I,B)", counts["isotope"], counts["elements"])
    print_step("Formal charge filter (-2 to +2)", counts["elements"], counts["charge"])
    print_step("MW filter (100 to 650)", counts["charge"], counts["mw"])
    print_step("Deduplicate canonical SMILES", counts["mw"], len(canonical_smiles))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if not args.no_header:
            handle.write("smiles\n")
        for smiles in sorted(canonical_smiles):
            handle.write(f"{smiles}\n")

    print(f"Output written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
