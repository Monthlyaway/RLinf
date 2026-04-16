# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a small potential network from preference JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rlinf.reward_relabel.frozen_potential import (
    PotentialMLP,
    collect_successful_endpoints,
    structure_regularizer,
    trajectory_return,
)


class PreferencePairDataset(Dataset):
    """Dataset backed by a JSONL file of winner/loser state sequences."""

    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]
        if not self.records:
            raise ValueError(f"No preference pairs found in {path}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def collate_pairs(batch: list[dict]) -> list[dict]:
    """Return the raw batch for per-trajectory processing."""
    return batch


def pair_losses(
    model: PotentialMLP,
    record: dict,
    *,
    device: torch.device,
    loss_type: str,
    lambda_struct: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total loss, BT loss, and structural loss for one pair."""
    winner_states = torch.as_tensor(
        record["winner_states"], dtype=torch.float32, device=device
    )
    loser_states = torch.as_tensor(
        record["loser_states"], dtype=torch.float32, device=device
    )

    winner_score = trajectory_return(model, winner_states)
    loser_score = trajectory_return(model, loser_states)
    bt_loss = -F.logsigmoid(winner_score - loser_score)

    struct_loss = 0.5 * (
        structure_regularizer(model, winner_states)
        + structure_regularizer(model, loser_states)
    )
    if loss_type == "bt_struct":
        total_loss = bt_loss + lambda_struct * struct_loss
    else:
        total_loss = bt_loss
    return total_loss, bt_loss.detach(), struct_loss.detach()


def evaluate(
    model: PotentialMLP,
    dataset: PreferencePairDataset,
    *,
    device: torch.device,
    loss_type: str,
    lambda_struct: float,
) -> dict[str, float]:
    """Evaluate the model on a dataset of preference pairs."""
    model.eval()
    total = 0
    correct = 0
    total_bt_loss = 0.0
    total_struct_loss = 0.0

    with torch.inference_mode():
        for record in dataset.records:
            _, bt_loss, struct_loss = pair_losses(
                model,
                record,
                device=device,
                loss_type=loss_type,
                lambda_struct=lambda_struct,
            )
            winner_states = torch.as_tensor(
                record["winner_states"], dtype=torch.float32, device=device
            )
            loser_states = torch.as_tensor(
                record["loser_states"], dtype=torch.float32, device=device
            )
            prediction = trajectory_return(model, winner_states) > trajectory_return(
                model, loser_states
            )
            correct += int(prediction.item())
            total += 1
            total_bt_loss += float(bt_loss.item())
            total_struct_loss += float(struct_loss.item())

    pair_accuracy = correct / max(total, 1)
    return {
        "pair_accuracy": pair_accuracy,
        "pairwise_tau": 2.0 * pair_accuracy - 1.0,
        "bt_loss": total_bt_loss / max(total, 1),
        "struct_loss": total_struct_loss / max(total, 1),
    }


def export_normalization_stats(
    model: PotentialMLP,
    records: list[dict],
    *,
    device: torch.device,
    output_path: Path,
) -> dict[str, float]:
    """Export successful endpoint normalization statistics."""
    start_states, end_states = collect_successful_endpoints(records)
    if not start_states or not end_states:
        fallback_state = records[0]["winner_states"][0]
        start_states = [fallback_state]
        end_states = [records[0]["winner_states"][-1]]

    with torch.inference_mode():
        start_tensor = torch.as_tensor(start_states, dtype=torch.float32, device=device)
        end_tensor = torch.as_tensor(end_states, dtype=torch.float32, device=device)
        start_mean = float(model(start_tensor).mean().item())
        goal_mean = float(model(end_tensor).mean().item())

    normalization_stats = {
        "start_mean": start_mean,
        "goal_mean": goal_mean,
        "successful_episode_count": len(start_states),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(normalization_stats, file, indent=2)
    return normalization_stats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-type", choices=["bt_only", "bt_struct"], required=True)
    parser.add_argument("--lambda-struct", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Train the potential model and export checkpoint/metrics/stats."""
    args = parse_args()
    torch.manual_seed(args.seed)

    train_dataset = PreferencePairDataset(args.train_file)
    val_dataset = PreferencePairDataset(args.val_file)
    input_dim = len(train_dataset.records[0]["winner_states"][0])
    device = torch.device(args.device)

    model = PotentialMLP(
        input_dim=input_dim,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
        activation="tanh",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state_dict = None
    best_metrics = None
    best_val_accuracy = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_bt_loss = 0.0
        running_struct_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_losses = []
            batch_bt_losses = []
            batch_struct_losses = []
            for record in batch:
                total_loss, bt_loss, struct_loss = pair_losses(
                    model,
                    record,
                    device=device,
                    loss_type=args.loss_type,
                    lambda_struct=args.lambda_struct,
                )
                batch_losses.append(total_loss)
                batch_bt_losses.append(bt_loss)
                batch_struct_losses.append(struct_loss)

            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()

            running_total_loss += float(loss.item())
            running_bt_loss += float(torch.stack(batch_bt_losses).mean().item())
            running_struct_loss += float(torch.stack(batch_struct_losses).mean().item())
            running_batches += 1

        val_metrics = evaluate(
            model,
            val_dataset,
            device=device,
            loss_type=args.loss_type,
            lambda_struct=args.lambda_struct,
        )
        if val_metrics["pair_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["pair_accuracy"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch,
                "best_val_pair_accuracy": val_metrics["pair_accuracy"],
                "best_val_pairwise_tau": val_metrics["pairwise_tau"],
                "best_val_bt_loss": val_metrics["bt_loss"],
                "best_val_struct_loss": val_metrics["struct_loss"],
                "train_total_loss": running_total_loss / max(running_batches, 1),
                "train_bt_loss": running_bt_loss / max(running_batches, 1),
                "train_struct_loss": running_struct_loss / max(running_batches, 1),
            }

    if best_state_dict is None or best_metrics is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    checkpoint = {
        "model_state_dict": best_state_dict,
        "model_cfg": {
            "input_dim": input_dim,
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "activation": "tanh",
            "loss_type": args.loss_type,
        },
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=2)

    model.load_state_dict(best_state_dict)
    export_normalization_stats(
        model,
        train_dataset.records,
        device=device,
        output_path=args.output_dir / "normalization_stats.json",
    )


if __name__ == "__main__":
    main()
