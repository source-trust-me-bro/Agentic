"""REST client for the application backend's topic-graph endpoints.

The backend (KalevaShorts API) owns the topic graph: schema, validation and
bidirectional persistence. The newsroom only reads topics and submits
relationships through its documented REST API — it never touches that database.

Documented contract (Backend README, "Topic relationships"; base URL must point
at the API root, e.g. ``http://127.0.0.1:8000``):

* ``GET  {base}/api/v1/topics/tree``  (reader) — the topic tree. Every node
  carries ``branches``: the ids of topics it is semantically related to,
  stored symmetrically by the backend.
* ``POST {base}/api/v1/topics/{topic_id}/branches``  (editor) — body
  ``{"branch_id": "<other topic id>"}`` links two topics. Idempotent,
  symmetric, and self-links or unknown ids are rejected (400 / 404).

Auth follows the backend's conventions: request berries a bearer token
(``Authorization: Bearer <Settings.backend_api_token>``) which the deployment
provisions from a credential the backend accepts with the editor role.
"""

from __future__ import annotations

from typing import Any

import httpx

from newsroom.config import Settings, get_settings

API_PREFIX = "/api/v1"


class BackendError(RuntimeError):
    """The backend rejected a call or could not be reached."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class BackendTimeout(BackendError):
    """The backend did not answer in time. Safe to retry: submissions are idempotent."""


class BackendClient:
    """Small stateless client. ``transport`` is injectable for hermetic tests."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("BackendClient needs a base_url.")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._timeout = timeout_seconds
        self._transport = transport

    # ------------------------------------------------------------------ topic reads

    def list_topics(self) -> list[dict[str, Any]]:
        """Flat list of topics from the backend tree, each with its ``branches``."""

        payload = self._get(API_PREFIX + "/topics/tree").json()
        topics: list[dict[str, Any]] = []
        seen: set[str] = set()

        def walk(nodes: Any, parent_name: str | None) -> None:
            for node in nodes or []:
                topic = node.get("topic", node)
                topic_id = str(topic.get("id", ""))
                if topic_id and topic_id not in seen:
                    seen.add(topic_id)
                    entry = dict(topic)
                    entry["id"] = topic_id
                    entry["branches"] = list(topic.get("branches") or [])
                    entry["category"] = parent_name
                    topics.append(entry)
                walk(node.get("children"), topic.get("name"))

        walk(payload, None)
        return topics

    # ------------------------------------------------------------------ topic links

    def link_topics(self, first_id: str, second_id: str) -> dict[str, Any]:
        """Submit one relationship. The backend validates and stores it symmetrically."""

        if not first_id or not second_id:
            raise BackendError("Both topic ids are required to submit a relationship.")
        if first_id == second_id:
            raise BackendError("A topic cannot be linked to itself.")
        response = self._post(f"{API_PREFIX}/topics/{first_id}/branches", {"branch_id": second_id})
        return response.json()

    # -------------------------------------------------------------------- plumbing

    def _get(self, path: str) -> httpx.Response:
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = client.request(
                    method, f"{self.base_url}{path}", json=body, headers=headers
                )
            except httpx.TimeoutException as exc:
                raise BackendTimeout(f"backend {method} {path} timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise BackendError(f"backend {method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            raise BackendError(
                f"backend {method} {path} -> {response.status_code}: {_detail(response)}",
                status=response.status_code,
            )
        return response


def _detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return (response.text or "")[:200]
    return str(data.get("detail", data))[:200]


def client_from_settings(settings: Settings | None = None) -> BackendClient | None:
    """The configured client, or None when the backend integration is switched off."""

    settings = settings or get_settings()
    if not settings.backend_api_url:
        return None
    return BackendClient(
        settings.backend_api_url,
        token=settings.backend_api_token,
        timeout_seconds=settings.http_timeout_seconds,
    )
