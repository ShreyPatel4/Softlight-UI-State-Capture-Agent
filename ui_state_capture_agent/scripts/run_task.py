import argparse
from src.agent.orchestrator import FlowSummary, run_task_query_blocking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language task query for Agent B")
    args = parser.parse_args()

    summary: FlowSummary = run_task_query_blocking(args.query)

    title_part = f" title={summary.task_title!r}" if summary.task_title else ""
    print(
        f"Flow finished: id={summary.id} app={summary.app_name} "
        f"run_id={summary.run_id} status={summary.status}{title_part}"
    )


if __name__ == "__main__":
    main()
