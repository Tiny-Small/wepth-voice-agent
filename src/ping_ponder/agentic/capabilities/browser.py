"""Browser capability: Local Jev goals SEARCH/NAVIGATE/BACK/FORWARD.

Adding this capability requires no change to Global Jev; it is registered like any
other descriptor.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from ping_ponder.observability import emit

from ..goals import GoalSchema, SlotSpec
from ..operators import ActionOperator, OperatorError, OperatorMetadata
from ..registry import CapabilityDescriptor
from ..world import Arg, Compare, Condition, Effect, WorldState
from ..adapters import BrowserAdapter, MemoryBrowserAdapter

logger = logging.getLogger(__name__)

WORLD_SCHEMA: Mapping[str, Any] = {
    "browser.running": False,
    "browser.current_url": None,
    "browser.current_domain": None,
    "browser.search_results": None,
    "browser.search_query": None,
    "browser.navigation_target": None,
    "browser.history": (),
    "browser.history_index": -1,
    "browser.can_back": False,
    "browser.can_forward": False,
    "browser.last_task_status": None,
    "browser.last_task_goal_type": None,
    "browser.last_task_site": None,
    "browser.last_task_target": None,
    "browser.last_task_evidence": {},
}
BROWSER_WORLD_SCHEMA = WORLD_SCHEMA

# Bare names are ambiguous input, so only explicitly reviewed aliases may be
# converted into destinations. Unknown names must be provided as a full domain/URL.
_TRUSTED_SITE_URLS = {
    "youtube": "https://www.youtube.com/",
    "google": "https://www.google.com/",
    "github": "https://github.com/",
    "hugging face": "https://huggingface.co/",
    "huggingface": "https://huggingface.co/",
    "arxiv": "https://arxiv.org/",
    "wikipedia": "https://www.wikipedia.org/",
}


def _resolved_navigation_target(world: WorldState, arguments: Mapping[str, Any]) -> str:
    """Project the same destination that the runtime adapter will navigate to."""
    return resolve_navigation_target(arguments["target"])


def _history_entry(world: WorldState, index: int) -> Any:
    history = tuple(world.get("browser.history") or ())
    return history[index] if 0 <= index < len(history) else world.get("browser.current_url")


def _has_back_history(world: WorldState) -> bool:
    return bool(world.get("browser.can_back")) or int(world.get("browser.history_index") or 0) > 0


def _has_forward_history(world: WorldState) -> bool:
    if world.get("browser.can_forward"):
        return True
    index = int(world.get("browser.history_index") if world.get("browser.history_index") is not None else -1)
    return index + 1 < len(tuple(world.get("browser.history") or ()))


def resolve_navigation_target(value: Any) -> str:
    """Resolve a trusted bare site alias or validate an explicit HTTP(S) target."""
    text = str(value or "").strip()
    if not text:
        raise OperatorError("Navigation target is empty")

    alias = text.casefold().strip(".,!?;:")
    if alias in _TRUSTED_SITE_URLS:
        return _TRUSTED_SITE_URLS[alias]

    has_scheme = "://" in text
    candidate = text if has_scheme else f"https://{text}"
    try:
        parsed = urlsplit(candidate)
    except ValueError as error:
        raise OperatorError(f"Invalid navigation URL {text!r}: {error}") from error
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise OperatorError("Navigation only supports HTTP or HTTPS URLs")
    if parsed.username is not None or parsed.password is not None:
        raise OperatorError("Navigation URLs must not contain embedded credentials")
    host = parsed.hostname
    if not host or any(character.isspace() for character in host):
        raise OperatorError(
            f"Unrecognized bare site name {text!r}; use a trusted site name or provide its domain"
        )
    try:
        parsed.port  # Validate malformed port syntax before passing to the adapter.
    except ValueError as error:
        raise OperatorError(f"Invalid navigation URL {text!r}: {error}") from error
    if "." not in host:
        raise OperatorError(
            f"Unrecognized bare site name {text!r}; use a trusted site name or provide its domain"
        )
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, parsed.path or "/",
                       parsed.query, parsed.fragment))


def _is_navigation_destination(value: str) -> bool:
    """Accept only a domain, URL, or reviewed bare-site alias as NAVIGATE input."""
    try:
        resolve_navigation_target(value)
    except OperatorError:
        return False
    return True


def _web_url_parts(value: Any) -> tuple[str, str, str] | None:
    """Return normalized host/path/query for an HTTP(S) URL or bare domain."""
    text = str(value or "").strip()
    if not text:
        return None
    has_scheme = "://" in text
    try:
        parsed = urlsplit(text if has_scheme else f"//{text}")
    except ValueError:
        return None
    if parsed.scheme and parsed.scheme.casefold() not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    default_port = 443 if parsed.scheme.casefold() == "https" else 80
    if port is not None and port != default_port:
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return host, path, parsed.query


def _navigation_matches_target(world: WorldState, arguments: Mapping[str, Any]) -> bool:
    """Match observed state to the trusted, explicit destination for a goal.

    Bare names are resolved through the trusted alias table; a search-results URL
    whose query merely contains the name is not considered a match.
    """
    target = str(arguments.get("target") or "").strip()
    current_url = world.get("browser.current_url")
    actual = _web_url_parts(current_url)
    if not target or actual is None:
        return False
    try:
        resolved_target = resolve_navigation_target(target)
    except OperatorError:
        return False
    expected = _web_url_parts(resolved_target)
    if expected is None:
        return False
    hosts_match = actual[0] == expected[0]
    paths_match = expected[1] == "/" or actual[1] == expected[1]
    queries_match = not expected[2] or actual[2] == expected[2]
    return hosts_match and paths_match and queries_match


def browser_goal_types() -> tuple[str, ...]:
    """Goal types are declared once, in `browser_goal_schemas`; this is a convenience view."""
    return tuple(browser_goal_schemas())


def browser_operators(adapter: BrowserAdapter | None = None, *, browser_find: Any | None = None
                      ) -> tuple[ActionOperator, ...]:
    adapter = adapter or MemoryBrowserAdapter(WORLD_SCHEMA)

    async def open_adapter(world, arguments):
        await adapter.open()
        return {}

    async def search_adapter(world, arguments):
        await adapter.search(arguments["query"])
        return {}

    async def navigate_adapter(world, arguments):
        requested = str(arguments["target"])
        resolved = resolve_navigation_target(requested)
        emit(logger, "browser_navigation_target_resolved", requested=requested,
             resolved=resolved, trusted_alias=requested.casefold().strip(".,!?;:") in _TRUSTED_SITE_URLS)
        await adapter.navigate(resolved)
        return {}

    async def back_adapter(world, arguments):
        await adapter.back()
        return {}

    async def forward_adapter(world, arguments):
        await adapter.forward()
        return {}

    operators = [
        ActionOperator(
            name="OpenBrowser",
            preconditions=(Condition("browser.running", Compare.FALSY),),
            effects=(Effect("browser.running", True),),
            cost=1.0,
            executor=open_adapter,
            satisfies_goals=("OPEN",),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="SearchWeb",
            parameters=("query",),
            preconditions=(Condition("browser.running"),),
            # Declared effects must match exactly what the executor observes.
            effects=(Effect("browser.search_results", Arg("query")),
                     Effect("browser.search_query", Arg("query")),
                     Effect("browser.current_url", derive=lambda world, args: f"search://{args['query']}"),
                     Effect("browser.history", derive=lambda world, args: (*tuple(world.get("browser.history") or ()), f"search://{args['query']}")),
                     Effect("browser.history_index", derive=lambda world, args: len(tuple(world.get("browser.history") or ())))),
            cost=1.0,
            executor=search_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=True, idempotent=False),
        ),
        ActionOperator(
            name="NavigateURL",
            parameters=("target",),
            preconditions=(Condition("browser.running"),),
            effects=(Effect("browser.current_url", derive=_resolved_navigation_target),
                     Effect("browser.navigation_target", derive=_resolved_navigation_target),
                     Effect("browser.history", derive=lambda world, args: (
                         *tuple(world.get("browser.history") or ()),
                         _resolved_navigation_target(world, args),
                     )),
                     Effect("browser.history_index", derive=lambda world, args: len(tuple(world.get("browser.history") or ())))),
            cost=1.0,
            executor=navigate_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="Back",
            preconditions=(Condition("browser.history_index", predicate=_has_back_history),),
            satisfies_goals=("BACK",),
            effects=(Effect("browser.history_index", derive=lambda world, args: max(0, int(world.get("browser.history_index") or 0) - 1)),
                     Effect("browser.current_url", derive=lambda world, args: _history_entry(world, int(world.get("browser.history_index") or 0) - 1))),
            cost=1.0,
            executor=back_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=False, idempotent=False),
        ),
        ActionOperator(
            name="Forward",
            # Index bound is a cross-key predicate the declarative vocabulary cannot state.
            preconditions=(
                Condition("browser.history_index", predicate=_has_forward_history),
            ),
            satisfies_goals=("FORWARD",),
            effects=(Effect("browser.history_index", derive=lambda world, args: int(world.get("browser.history_index") or 0) + 1),
                     Effect("browser.current_url", derive=lambda world, args: _history_entry(world, int(world.get("browser.history_index") or 0) + 1))),
            cost=1.0,
            executor=forward_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=False, idempotent=False),
        ),
    ]
    if browser_find is not None:
        async def run_grounded(world, arguments, goal_type):
            from ..goals import SemanticGoal

            target_name = "query" if goal_type == "SEARCH_WEBSITE" else "target"
            goal = SemanticGoal(
                capability="Browser", goal_type=goal_type,
                arguments={name: arguments[name] for name in ("site", target_name) if name in arguments},
            )
            emit(logger, "browser_find_handoff", goal=goal.describe())
            result, _ = await browser_find.execute(goal, world)
            changes = result.world_changes()
            changes["browser.last_task_goal_type"] = goal_type
            changes["browser.last_task_site"] = arguments.get("site")
            changes["browser.last_task_target"] = arguments[target_name]
            emit(logger, "browser_find_result", status=result.completion.status.value,
                 evidence=changes["browser.last_task_evidence"])
            keys = (
                "browser.running", "browser.current_url", "browser.current_domain",
                "browser.last_task_status", "browser.last_task_goal_type",
                "browser.last_task_site", "browser.last_task_target",
                "browser.last_task_evidence",
            )
            return {key: changes[key] for key in keys}

        async def find_with_grounding(world, arguments):
            return await run_grounded(world, arguments, "FIND")

        async def search_website_with_grounding(world, arguments):
            return await run_grounded(world, arguments, "SEARCH_WEBSITE")

        operators.append(ActionOperator(
            name="FindWithGroundedBrowser",
            parameters=("target",),
            optional_parameters=("site",),
            effects=(
                Effect("browser.running", True),
                Effect("browser.current_url", derive=lambda world, args: world.get("browser.current_url")),
                Effect("browser.current_domain", derive=lambda world, args: world.get("browser.current_domain")),
                Effect("browser.last_task_status", "satisfied"),
                Effect("browser.last_task_goal_type", "FIND"),
                Effect("browser.last_task_site", derive=lambda world, args: args.get("site")),
                Effect("browser.last_task_target", Arg("target")),
                Effect("browser.last_task_evidence", derive=lambda world, args: world.get("browser.last_task_evidence")),
            ),
            cost=1.0,
            executor=find_with_grounding,
            metadata=OperatorMetadata(
                self_observing=True, terminal_on_execution=True, idempotent=False,
            ),
        ))
        operators.append(ActionOperator(
            name="SearchWebsiteWithGroundedBrowser",
            parameters=("site", "query"),
            effects=(
                Effect("browser.running", True),
                Effect("browser.current_url", derive=lambda world, args: world.get("browser.current_url")),
                Effect("browser.current_domain", derive=lambda world, args: world.get("browser.current_domain")),
                Effect("browser.last_task_status", "satisfied"),
                Effect("browser.last_task_goal_type", "SEARCH_WEBSITE"),
                Effect("browser.last_task_site", Arg("site")),
                Effect("browser.last_task_target", Arg("query")),
                Effect("browser.last_task_evidence", derive=lambda world, args: world.get("browser.last_task_evidence")),
            ),
            cost=1.0,
            executor=search_website_with_grounding,
            metadata=OperatorMetadata(
                self_observing=True, terminal_on_execution=True, idempotent=False,
            ),
        ))
    return tuple(operators)


def browser_goal_schemas() -> Mapping[str, GoalSchema]:
    return {
        "OPEN": GoalSchema("OPEN", satisfied_when=(Condition("browser.running"),)),
        "SEARCH": GoalSchema("SEARCH", slots=(SlotSpec("query", "What does the user want to search for?"),),
                             satisfied_when=(Condition("browser.search_results", Compare.EQ, Arg("query")),)),
        "SEARCH_WEBSITE": GoalSchema(
            "SEARCH_WEBSITE",
            intent_description=("Search within a named website and stop on its search results page; "
                                "do not open an individual result"),
            slots=(
                SlotSpec("site", "Which website should be searched?"),
                SlotSpec("query", "What should be entered into that website's search?"),
            ),
            satisfied_when=(
                Condition("browser.last_task_status", Compare.EQ, "satisfied"),
                Condition("browser.last_task_goal_type", Compare.EQ, "SEARCH_WEBSITE"),
                Condition("browser.last_task_site", Compare.EQ, Arg("site")),
                Condition("browser.last_task_target", Compare.EQ, Arg("query")),
            ),
        ),
        "NAVIGATE": GoalSchema(
            "NAVIGATE",
            intent_description=("Open a specific URL, domain, or trusted site alias supplied by the user"),
            slots=(SlotSpec(
                "target", "What URL, domain, or trusted site name should be opened?",
                validator=_is_navigation_destination,
            ),),
                               satisfied_when=(Condition(
                                   "browser.current_url",
                                   bound_predicate=_navigation_matches_target,
                               ),)),
        "FIND": GoalSchema(
            "FIND",
            intent_description=("Locate a page, item, or information within a website using page understanding "
                                "and grounded page interactions; target is a human-readable target"),
            slots=(
                SlotSpec("site", "Which website's content should be searched?", required=False),
                SlotSpec("target", "What page, item, or information should be found?"),
            ),
            satisfied_when=(
                Condition("browser.last_task_status", Compare.EQ, "satisfied"),
                Condition("browser.last_task_target", Compare.EQ, Arg("target")),
            ),
        ),
        "BACK": GoalSchema("BACK", command=True),
        "FORWARD": GoalSchema("FORWARD", command=True),
    }


class BrowserCapability:
    """Bundles Browser Local Jev goals, schemas, and operators."""

    name = "Browser"
    description = ("Web browsing: search the web, navigate to a site or URL, find a page or item using "
                   "grounded page reasoning, and go back or forward in history.")
    world_schema = WORLD_SCHEMA

    def __init__(self, local_jev: Any, adapter: BrowserAdapter | None = None, *,
                 browser_find: Any | None = None) -> None:
        self.local_jev = local_jev
        self.adapter = adapter or MemoryBrowserAdapter(WORLD_SCHEMA)
        self.browser_find = browser_find

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(name=self.name, description=self.description,
                                    local_jev=self.local_jev,
                                    goal_schemas=browser_goal_schemas(),
                                    operators=browser_operators(self.adapter, browser_find=self.browser_find),
                                    world_schema=WORLD_SCHEMA,
                                    observer=self.adapter.observe)


def build_browser_descriptor(local_jev: Any, *, adapter: BrowserAdapter | None = None,
                             browser_find: Any | None = None) -> CapabilityDescriptor:
    return BrowserCapability(local_jev, adapter=adapter, browser_find=browser_find).descriptor()
