import argparse

from src.agent.orchestrator import run_task_query_blocking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent-driven UI task")
    parser.add_argument("--query", required=True, help="Task query, e.g. 'linear: create task'")
    parser.add_argument(
        "--app",
        required=False,
        help="Optional app override (linear, notion, outlook). Overrides any prefix in the query.",
    )
    return parser.parse_args()


def build_query(query: str, app_override: str | None) -> str:
    if app_override:
        return f"{app_override}:{query.split(':', 1)[-1].strip()}"
    return query


def main() -> None:
    args = parse_args()
    normalized_query = build_query(args.query, args.app)
    flow = run_task_query_blocking(normalized_query)
    print(f"Flow completed: {flow.id} (status={flow.status})")


if __name__ == "__main__":
    main()
