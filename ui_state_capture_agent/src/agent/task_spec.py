from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """Lightweight representation of a natural language request."""

    raw_query: str
    app_name: str
    goal: str
    object_type: str
    constraints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_query(cls, raw_query: str) -> "TaskSpec":
        """Parse a raw natural language query into a ``TaskSpec``.

        The parsing logic is intentionally simple for now, expecting queries like::

            "linear: create project for TES"

        The text before the first ``:`` is treated as the app name and the
        remainder as the goal.
        """

        app_name, _, goal_text = raw_query.partition(":")
        return cls(
            raw_query=raw_query,
            app_name=app_name.strip(),
            goal=goal_text.strip(),
            object_type="",
            constraints={},
        )
