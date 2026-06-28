"""
추천 모델 학습 스크립트.
사용:
    uv run python scripts/train.py --model mf
    uv run python scripts/train.py --model ncf
    uv run python scripts/train.py --model sasrec
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from models.evaluate import evaluate
from models.mf import BPRDataset, MatrixFactorization
from models.ncf import NeuMF
from models.sasrec import SASRec, SASRecDataset

DATA_DIR = Path("data/processed/All_Beauty")
MAX_LEN  = 50


def build_model(model: str, n_users: int, n_items: int, n_factors: int):
    if model == "mf":
        return MatrixFactorization(n_users, n_items, n_factors)
    if model == "ncf":
        return NeuMF(n_users, n_items, n_factors, mlp_layers=[64, 32, 16])
    if model == "sasrec":
        return SASRec(n_items, hidden_dim=n_factors, max_len=MAX_LEN, n_heads=2, n_layers=2)
    raise ValueError(f"Unknown model: {model}")


def make_sasrec_score_fn(model, train_df, max_len):
    """SASRec 전용 score_fn: 유저 시퀀스를 구성해서 마지막 위치로 점수 계산."""
    user_seqs = {
        uid: [i + 1 for i in grp.sort_values("timestamp")["parent_asin"].tolist()]
        for uid, grp in train_df.groupby("user_id")
    }

    def _pad(seq):
        seq = seq[-max_len:]
        return [0] * (max_len - len(seq)) + seq

    def score_fn(user_id, item_ids):
        seq = torch.tensor(_pad(user_seqs[user_id]), dtype=torch.long)
        items = torch.tensor([i + 1 for i in item_ids], dtype=torch.long)  # +1 shift
        return model.score_seq(seq, items)

    return score_fn


def run(
    model: str = "mf",
    n_factors: int = 32,
    lr: float = 1e-3,
    batch_size: int = 512,
    n_epochs: int = 50,
    K: int = 10,
) -> tuple:
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")
    with open(DATA_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    n_users, n_items = meta["n_users"], meta["n_items"]

    # Dataset
    if model == "sasrec":
        dataset = SASRecDataset(train_df, n_items, max_len=MAX_LEN)
    else:
        dataset = BPRDataset(train_df, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mdl       = build_model(model, n_users, n_items, n_factors)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)

    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"[{model.upper()}] params={n_params:,}, n_factors={n_factors}")
    print(f"{'Epoch':>6} {'Loss':>8} {f'Hit@{K}':>8} {f'NDCG@{K}':>9}")
    print("-" * 36)

    history = []
    for epoch in range(1, n_epochs + 1):
        mdl.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = mdl(*batch) if model == "sasrec" else mdl.bpr_loss(*batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 10 == 0:
            mdl.eval()
            score_fn = (
                make_sasrec_score_fn(mdl, train_df, MAX_LEN)
                if model == "sasrec"
                else (lambda uid, iids: mdl(
                    torch.tensor([uid] * len(iids)),
                    torch.tensor(iids),
                ))
            )
            metrics = evaluate(score_fn, train_df, test_df, n_items, K=K)
            hit, ndcg = metrics[f"hit@{K}"], metrics[f"ndcg@{K}"]
            print(f"{epoch:>6} {avg_loss:>8.4f} {hit:>8.4f} {ndcg:>9.4f}")
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

    save_path = DATA_DIR / f"{model}_model.pt"
    torch.save(mdl.state_dict(), save_path)
    print(f"\n모델 저장 완료 → {save_path}")
    return mdl, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="mf",  choices=["mf", "ncf", "sasrec"])
    parser.add_argument("--n_factors",  default=32,    type=int)
    parser.add_argument("--lr",         default=1e-3,  type=float)
    parser.add_argument("--batch_size", default=512,   type=int)
    parser.add_argument("--n_epochs",   default=50,    type=int)
    parser.add_argument("--K",          default=10,    type=int)
    args = parser.parse_args()
    run(**vars(args))
