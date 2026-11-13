from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    workspace_id: str
    role: str
    request_id: str = ""
