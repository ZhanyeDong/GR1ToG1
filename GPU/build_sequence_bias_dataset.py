import argparse
import os
from typing import Iterable, List

import torch

from pinocchio_fk_ik_check import get_example_checker




def to_nx7_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        t = data.to(torch.float32)
        if t.ndim == 1:
            if t.numel() != 7:
                raise ValueError(f"Expected 7 values, got shape {tuple(t.shape)}")
            return t.unsqueeze(0)
        if t.ndim == 2 and t.shape[1] == 7:
            return t
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")

    if isinstance(data, (list, tuple)):
        rows = [torch.as_tensor(x, dtype=torch.float32) for x in data]
        if len(rows) == 0:
            return torch.empty((0, 7), dtype=torch.float32)
        stacked = torch.stack(rows, dim=0)
        if stacked.ndim == 2 and stacked.shape[1] == 7:
            return stacked
        raise ValueError(f"Unsupported list->tensor shape: {tuple(stacked.shape)}")

    raise TypeError(f"Unsupported loaded data type: {type(data)}")


def parse_bias_values(raw: str) -> torch.Tensor:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        v = float(parts[0])
        if v < 0:
            raise ValueError("bias_abs must be non-negative")
        return torch.full((7,), v, dtype=torch.float32)
    if len(parts) == 7:
        vals = [float(x) for x in parts]
        if any(v < 0 for v in vals):
            raise ValueError("bias_abs values must be non-negative")
        return torch.tensor(vals, dtype=torch.float32)
    raise ValueError("bias_abs must be one value or 7 comma-separated values")


def apply_bias_and_clip(
    sequence: torch.Tensor,
    bias_abs: torch.Tensor,
) -> torch.Tensor:
    bias = (torch.rand(sequence.shape) * 2.0 - 1.0) * bias_abs.unsqueeze(0)
    return sequence + bias


def build_dataset(
    train_files: Iterable[str],
    bias_abs: torch.Tensor,
) -> torch.Tensor:

    all_samples: List[torch.Tensor] = []
    for path in train_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        sequence = to_nx7_tensor(torch.load(path))
        # Always keep original sequence
        all_samples.append(sequence)

        biased_sequence = apply_bias_and_clip(
            sequence=sequence,
            bias_abs=bias_abs,
        )
        all_samples.append(biased_sequence)

    if not all_samples:
        raise RuntimeError("No data generated. Check input files and arguments.")

    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build training set from imported sequences + random per-sequence bias."
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        default=[
            "action_list0126_1.pt",
            "action_list0126_2.pt",
            "action_list0126_3.pt",
            "action_list0126_4.pt",
            "action_list0126_5.pt",
            "action_list0305_5.pt",
            "action_list0305_6.pt",
            "action_list0318_1.pt",
        ],
        help="Input sequence files (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_dataset_0322.pt",
        help="Output training tensor path (.pt).",
    )
    parser.add_argument(
        "--bias_abs",
        type=str,
        default="0.2",
        help="One value (all joints) or 7 comma values, e.g. 0.05 or 0.03,0.03,0.04,0.05,0.05,0.05,0.06",
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=1000000,
        help="If > 0, force final output to exactly this sample count.",
    )
    # bias_mode is fixed to "sample"
    # pad_mode is fixed to "sequence_random"
    # validate_g1_ik is always enabled
    # show_sample_count is always enabled
    # IK settings are embedded in pinocchio_fk_ik_check.example
    args = parser.parse_args()

    bias_abs = parse_bias_values(args.bias_abs)
    if args.target_samples < 0:
        raise ValueError("target_samples must be >= 0")

    target_n = int(args.target_samples) if args.target_samples > 0 else None
    dataset_samples: List[torch.Tensor] = []
    round_idx = 0
    checker = get_example_checker()
    while target_n is None or len(dataset_samples) < target_n:
        round_idx += 1
        added_this_round = 0

        file_order = torch.randperm(len(args.train_files)).tolist()
        for file_i in file_order:
            path = args.train_files[file_i]
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            sequence = to_nx7_tensor(torch.load(path))
            if round_idx == 1:
                blocks = (sequence,)
            else:
                biased_sequence = apply_bias_and_clip(
                    sequence=sequence,
                    bias_abs=bias_abs,
                )
                blocks = (biased_sequence,)

            for block in blocks:
                if isinstance(block, torch.Tensor):
                    block_cpu = block.detach().cpu().numpy()
                else:
                    block_cpu = block
                for i in range(block.shape[0]):
                    q_right_gr1 = block_cpu[i]
                    if checker.check(q_right_gr1):
                        dataset_samples.append(block[i])
                        added_this_round += 1
                        if target_n is not None and len(dataset_samples) >= target_n:
                            break
                if target_n is not None and len(dataset_samples) >= target_n:
                    break
            if target_n is not None and len(dataset_samples) >= target_n:
                break

        print(f"Round {round_idx}: {len(dataset_samples)} samples")

    dataset = torch.stack(dataset_samples, dim=0)

    torch.save(dataset, args.output)
    sample_count = int(dataset.shape[0])

    print(f"Saved: {args.output}")
    print(f"Sample count: {sample_count}")
    print(f"Bias abs: {bias_abs.tolist()}")
    print(f"Train files: {args.train_files}")
    print(sample_count)


if __name__ == "__main__":
    main()
