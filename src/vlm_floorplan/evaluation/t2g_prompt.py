"""What the text-to-graph model is sent: the training system prompt, the tools, the user turn.

The system prompt and the twelve tools are exactly what training uses (`agent.episode`,
`agent.tools`), so the base model and a fine-tuned one see the same request. The only variant is
a worked example appended to the system prompt, which shows how far prompting alone gets.
"""

from __future__ import annotations

from vlm_floorplan.agent.episode import SYSTEM_PROMPT
from vlm_floorplan.compact import to_compact
from vlm_floorplan.edits.summary import summarize_graph
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room
from vlm_floorplan.text2graph.rows import PromptRow

PROMPT_VERSION = "t2g-v1"
VARIANTS = ("as-is", "one-example")
EXAMPLE_REQUEST = (
    "A one-bedroom flat with a bathroom and a kitchen open to the living room."
    " The front door opens into the living room."
)
_EXAMPLE_REPLY = (
    "Drew a 4-room plan: 1 bedroom, 1 bathroom, the kitchen r2 open to the living room r3,"
    " the entrance into the living room r3."
)


def example_plan() -> FloorplanGraph:
    """A hand-made four-room plan, never taken from any dataset."""
    rooms = [
        Room(id="r0", type="bedroom", xy=(250, 250), size=130),
        Room(id="r1", type="bathroom", xy=(560, 250), size=70),
        Room(id="r2", type="kitchen", xy=(760, 560), size=110),
        Room(id="r3", type="living", xy=(380, 600), size=170, entrance=True),
    ]
    links = [
        Link(a="r0", b="r1", kind="wall"),
        Link(a="r0", b="r3", kind="door"),
        Link(a="r1", b="r3", kind="door"),
        Link(a="r2", b="r3", kind="opening"),
    ]
    return FloorplanGraph(
        floors=[Floor(floor=1, rooms=rooms, links=links)],
        image=ImageSize(w=1000, h=1000),
        exterior_wall=[(60, 150), (60, 800), (900, 800), (900, 150)],
    )


def user_message(start: FloorplanGraph, request: str) -> str:
    """The user turn of a sample: the current plan with its exterior wall, then the request."""
    return f"Current plan:\n{summarize_graph(start)}\n\nRequest: {request}"


def system_prompt(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "as-is":
        return SYSTEM_PROMPT
    plan = example_plan()
    empty = plan.model_copy(deep=True)
    empty.floors[0].rooms, empty.floors[0].links = [], []
    return (
        f"{SYSTEM_PROMPT}\n\nExample.\nUser message:\n{user_message(empty, EXAMPLE_REQUEST)}\n"
        f"You call new_plan with plan = {to_compact(plan)}\n"
        f"The tool answers OK and shows the plan. You then reply: {_EXAMPLE_REPLY}"
    )


def build_messages(row: PromptRow, variant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(variant)},
        {"role": "user", "content": user_message(row.start_graph(), row.text)},
    ]
