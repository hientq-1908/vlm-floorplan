import json
from pathlib import Path

import pytest
from t2g_fixtures import make_record, write_split

from vlm_floorplan.agent.tools import agent_tool_schemas
from vlm_floorplan.compact import to_compact
from vlm_floorplan.evaluation.client import ChatResponse, parse_chat_response
from vlm_floorplan.evaluation.t2g_harness import T2gConfig, build_request, collect_outputs
from vlm_floorplan.evaluation.t2g_prompt import system_prompt, user_message
from vlm_floorplan.evaluation.t2g_score_one import reject_reason
from vlm_floorplan.evaluation.t2g_scoring import score_run
from vlm_floorplan.requirements import apply_labels
from vlm_floorplan.text2graph.build import BuildConfig, build_sets
from vlm_floorplan.text2graph.pools import SourceFiles
from vlm_floorplan.text2graph.rows import read_references, read_rows


@pytest.fixture(scope="module")
def sets(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("t2g")
    directory = tmp / "resplan"
    directory.mkdir()
    for split, base in (("train", 0), ("val", 1000), ("test", 2000)):
        write_split(
            directory / f"{split}.jsonl",
            [make_record(base + i, "resplan", split) for i in range(60)],
        )
    out = tmp / "sets"
    build_sets(
        BuildConfig(
            out_dir=out,
            files=SourceFiles(resplan_dir=directory),
            quotas={"resplan": 30},
            behavior_quotas={"edit": 6, "restart": 3, "contradiction": 3, "unsupported": 3},
            log=lambda _message: None,
        )
    )
    return out


class GoldClient:
    """Answers each draw prompt with the source plan, and each behavior prompt correctly."""

    def __init__(self, rows, references, mode="gold"):
        self.rows, self.references, self.mode = rows, references, mode
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        user = request["messages"][-1]["content"]
        row = next(r for r in self.rows if user.endswith(r.text))
        if self.mode == "silent":
            return ChatResponse("I cannot help.", "", "stop", 1, 1, 0.0)
        if self.mode == "xml":
            graph = apply_labels(self.references[row.record_id].graph, row.parsed_requirements())
            xml = f"<tool_call><function=new_plan><parameter=plan>{to_compact(graph)}</parameter>"
            return ChatResponse("", xml, "stop", 1, 1, 0.0)
        if self.mode == "no_size":
            plan = json.dumps(
                {"rooms": [{"id": "r0", "type": "living", "xy": [5, 5]}], "links": []}
            )
            call = {"name": "new_plan", "arguments": json.dumps({"plan": plan})}
            return ChatResponse("", "", "tool_calls", 1, 1, 0.0, tool_calls=[call])
        if row.behavior in ("draw", "restart"):
            graph = apply_labels(self.references[row.record_id].graph, row.parsed_requirements())
            call = {"name": "new_plan", "arguments": json.dumps({"plan": to_compact(graph)})}
            return ChatResponse("", "", "tool_calls", 1, 1, 0.0, tool_calls=[call])
        if row.behavior == "edit":
            e = row.expect
            if e["edit"] == "link":
                args = {"room_a": e["a"], "room_b": e["b"], "kind": e["link"]}
                call = {"name": "set_link_kind", "arguments": json.dumps(args)}
            else:
                new = {"type": e["type"], "x": 500, "y": 500, "size": 40}
                call = {"name": "add_room", "arguments": json.dumps(new)}
            return ChatResponse("", "", "tool_calls", 1, 1, 0.0, tool_calls=[call])
        words = [group[0] for group in row.expect["groups"]]
        return ChatResponse(f"Do you mean {words[0]} or {words[-1]}?", "", "stop", 1, 1, 0.0)


def run(sets, name, mode, tmp_path, limit=None):
    rows = read_rows(sets / f"{name}.jsonl")
    config = T2gConfig(
        run_name=mode, prompts=str(sets / f"{name}.jsonl"), model="m", concurrency=2, limit=limit
    )
    client = GoldClient(rows, read_references(sets), mode)
    run_dir = tmp_path / f"{name}-{mode}"
    scored_rows = collect_outputs(config, client, run_dir)
    return score_run(run_dir, scored_rows), run_dir, client


def test_the_request_is_the_training_prompt_with_the_twelve_tools(sets):
    row = read_rows(sets / "template_test.jsonl")[0]
    request = build_request(T2gConfig(run_name="r", prompts="p", model="m"), row)
    assert request["tools"] == agent_tool_schemas() and len(request["tools"]) == 12
    system, user = request["messages"]
    assert system["content"] == system_prompt("as-is")
    assert user["content"] == user_message(row.start_graph(), row.text)
    assert user["content"].startswith("Current plan:\nplan: empty\nexterior wall: (")
    assert request["chat_template_kwargs"] == {"enable_thinking": True}


def test_the_example_variant_adds_a_worked_new_plan_call():
    assert system_prompt("one-example").startswith(system_prompt("as-is"))
    assert "You call new_plan with plan = " in system_prompt("one-example")
    with pytest.raises(ValueError):
        system_prompt("two-examples")


def test_a_model_that_returns_the_source_plan_scores_perfectly(sets, tmp_path):
    summary, run_dir, _ = run(sets, "template_test", "gold", tmp_path)
    assert summary["tool_call_rate"] == 1.0 and summary["validity_rate"] == 1.0
    assert summary["strict"]["all_met_rate"] == 1.0
    assert summary["strict"]["requirement_satisfaction"] == 1.0
    assert summary["l4_plan_score_incl_invalid"] == pytest.approx(1.0)
    assert summary["plausibility"]["n_plans"] == summary["n_prompts"]
    assert (run_dir / "metrics.json").exists() and (run_dir / "MANIFEST.json").exists()
    assert "inputs.jsonl" in {p.name for p in run_dir.iterdir()}


def test_a_reply_without_a_tool_call_is_counted_as_no_plan(sets, tmp_path):
    summary, _, _ = run(sets, "template_test", "silent", tmp_path)
    assert summary["tool_call_rate"] == 0.0 and summary["validity_rate"] == 0.0
    assert summary["strict"]["all_met_rate"] == 0.0
    assert summary["strict"]["requirement_satisfaction"] == 0.0
    assert summary["plausibility"]["n_plans"] == 0


def test_a_call_the_tool_refuses_is_counted_by_reason(sets, tmp_path):
    summary, _, _ = run(sets, "template_test", "no_size", tmp_path)
    assert summary["tool_call_rate"] == 1.0 and summary["validity_rate"] == 0.0
    assert summary["invalid_reasons"] == {"missing_size": summary["n_draw_prompts"]}


def test_a_tool_call_written_in_the_reasoning_counts_only_for_the_lenient_reading(sets, tmp_path):
    summary, _, _ = run(sets, "template_test", "xml", tmp_path)
    assert summary["tool_call_rate"] == 0.0 and summary["validity_rate"] == 0.0
    assert summary["lenient_plan_rate"] == 1.0
    assert summary["strict"]["all_met_rate"] == 0.0 and summary["lenient"]["all_met_rate"] == 1.0
    assert summary["plausibility_incl_lenient"]["n_plans"] == summary["n_prompts"]


def test_behavior_prompts_are_scored_by_what_the_model_did(sets, tmp_path):
    summary, _, _ = run(sets, "behavior_test", "gold", tmp_path)
    behavior = summary["behavior"]
    assert behavior["contradiction"]["correct_rate"] == 1.0
    assert behavior["unsupported"]["correct_rate"] == 1.0
    assert behavior["edit_door"]["edit_effect_ok_rate"] == 1.0
    assert summary["strict"]["all_met_prompts"] >= 1  # the restart prompts are scored as draws
    silent, _, _ = run(sets, "behavior_test", "silent", tmp_path)
    assert silent["behavior"]["edit_door"]["correct_rate"] == 0.0
    assert silent["behavior"]["contradiction"]["correct_rate"] == 0.0


def test_a_run_resumes_and_skips_answered_prompts(sets, tmp_path):
    _, run_dir, client = run(sets, "template_test", "gold", tmp_path, limit=5)
    assert len(client.requests) == 5
    rows = read_rows(sets / "template_test.jsonl")
    config = T2gConfig(
        run_name="gold", prompts=str(sets / "template_test.jsonl"), model="m", limit=8
    )
    again = GoldClient(rows, read_references(sets))
    collect_outputs(config, again, run_dir)
    assert len(again.requests) == 3
    with pytest.raises(ValueError):
        collect_outputs(config.model_copy(update={"temperature": 0.1}), again, run_dir)


def test_the_http_client_keeps_structured_tool_calls():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "new_plan", "arguments": '{"plan": "{}"}'}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    reply = parse_chat_response(body, 0.1)
    assert reply.tool_calls == [{"name": "new_plan", "arguments": '{"plan": "{}"}'}]
    assert parse_chat_response({"choices": [{"message": {"content": "hi"}}]}, 0).tool_calls == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "REJECTED (invalid_plan): The plan is not valid: it is not JSON (x at character 1).",
            "invalid_json",
        ),
        ('REJECTED (invalid_plan): room r0 has no "size": every room needs one.', "missing_size"),
        ("REJECTED (invalid_plan): The plan has no rooms.", "no_rooms"),
        ("REJECTED (invalid_plan): something odd", "other_invalid_plan"),
    ],
)
def test_reject_reason(text, reason):
    assert reject_reason(text) == reason
