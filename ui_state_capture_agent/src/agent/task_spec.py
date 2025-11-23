from dataclasses import dataclass, field

from .app_resolver import AppResolver, AppResolution


@dataclass
class TaskSpec:
    """Lightweight representation of a natural language request."""

    original_query: str
    app_name: str
    goal: str
    start_url: str
    object_type: str = ""
    constraints: dict[str, str] = field(default_factory=dict)
    known_app: bool = True

    def __post_init__(self) -> None:
        self.app_name = self.app_name.lower()

    @classmethod
    def from_query(cls, raw_query: str) -> "TaskSpec":
        return parse_task_query(raw_query)


_resolver = AppResolver()


def parse_task_query(raw_query: str) -> TaskSpec:
    resolution: AppResolution = _resolver.resolve(raw_query)
    return TaskSpec(
        original_query=raw_query,
        app_name=resolution.app_name.lower(),
        goal=resolution.normalized_goal,
        start_url=resolution.start_url,
        object_type="",
        constraints={},
        known_app=resolution.known,
    )
