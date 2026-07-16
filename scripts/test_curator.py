"""
Gemini 큐레이션 모듈 동작 테스트.
.env 파일에 GEMINI_API_KEY를 설정한 뒤 실행:

    uv run python scripts/test_curator.py
    uv run python scripts/test_curator.py --user_id 5 --query "향이 없는 보습 로션"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

console = Console()


def run(user_id: int = 0, query: str = "건성 피부에 좋은 수분 크림 추천해줘", top_k: int = 5):
    from llm.pipeline import RecommendationPipeline

    console.rule("[bold blue]Recommendation Pipeline Test")
    console.print(f"[bold]User ID:[/bold] {user_id}")
    console.print(f"[bold]Query:[/bold]   {query}\n")

    with console.status("파이프라인 로딩 중..."):
        pipeline = RecommendationPipeline.load()

    with console.status("① 쿼리 이해 중 (Gemini)..."):
        ctx, curated = pipeline.recommend(user_id=user_id, query=query, top_k=top_k)

    # 쿼리 이해 결과
    console.print(Panel(
        f"[bold]Intent:[/bold]      {ctx.intent}\n"
        f"[bold]Preferences:[/bold] {', '.join(ctx.preferences) or '-'}\n"
        f"[bold]Constraints:[/bold] {', '.join(ctx.constraints) or '-'}\n"
        f"[bold]Keywords:[/bold]    {', '.join(ctx.keywords) or '-'}",
        title="① Query Understanding",
        border_style="cyan",
    ))

    # 큐레이션 결과
    table = Table(title="② Curated Recommendations", show_lines=True)
    table.add_column("Rank",    style="bold", width=5)
    table.add_column("Title",   width=40)
    table.add_column("Rating",  width=8)
    table.add_column("Price",   width=10)
    table.add_column("DL Rank", width=8)
    table.add_column("Reason",  width=45)

    for i, item in enumerate(curated, start=1):
        rating_str = f"{item.avg_rating:.1f}★" if item.avg_rating else "-"
        table.add_row(
            str(i),
            item.title[:38],
            rating_str,
            item.price or "-",
            str(item.dl_rank),
            item.reason,
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, default=0)
    parser.add_argument("--query", type=str, default="건성 피부에 좋은 수분 크림 추천해줘")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    run(user_id=args.user_id, query=args.query, top_k=args.top_k)
