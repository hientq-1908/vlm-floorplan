import json
import re
from pathlib import Path

import pytest
from t2g_fixtures import make_record, write_split

from vlm_floorplan.agent.env import user_message
from vlm_floorplan.agent.tools import NEW_PLAN, agent_tool_schemas, call_tool
from vlm_floorplan.compact import to_compact
from vlm_floorplan.evaluation.client import ApiError, ChatResponse, parse_chat_response
from vlm_floorplan.evaluation.t2g_harness import T2gConfig, build_request, collect_outputs
from vlm_floorplan.evaluation.t2g_prompt import example_plan, system_prompt
from vlm_floorplan.evaluation.t2g_score_one import reject_reason
from vlm_floorplan.evaluation.t2g_scoring import score_run
from vlm_floorplan.requirements import all_met, apply_labels
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


def reply(content="", calls=(), finish="stop"):
    calls = [
        {"id": f"c{i}", "name": name, "arguments": json.dumps(args)}
        for i, (name, args) in enumerate(calls)
    ]
    return ChatResponse(content, "", "tool_calls" if calls else finish, 1, 1, 0.0, tool_calls=calls)


class GoldClient:
    """Answers each prompt correctly, one tool call per turn, reading the conversation so far."""

    def __init__(self, rows, references, mode="gold"):
        self.rows, self.references, self.mode = rows, references, mode
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        messages = request["messages"]
        user = messages[1]["content"]
        row = next(r for r in self.rows if user.endswith(r.text))
        made = [m["tool_calls"][0]["function"]["name"] for m in messages if m.get("tool_calls")]
        if self.mode == "silent":
            return reply("I cannot help.")
        if self.mode == "xml":
            graph = apply_labels(self.references[row.record_id].graph, row.parsed_requirements())
            xml = f"<tool_call><function=new_plan><parameter=plan>{to_compact(graph)}</parameter>"
            return ChatResponse("", xml, "stop", 1, 1, 0.0)
        if self.mode == "no_size":
            plan = json.dumps(
                {"rooms": [{"id": "r0", "type": "living", "xy": [5, 5]}], "links": []}
            )
            return reply(calls=[] if made else [(NEW_PLAN, {"plan": plan})])
        if made and (row.behavior != "edit" or row.expect["edit"] == "link" or len(made) == 2):
            return reply("Done.")
        if row.behavior == "restart" and self.mode == "keep":
            return reply(calls=[(NEW_PLAN, {"plan": to_compact(row.start_graph())})])
        if row.behavior in ("draw", "restart"):
            graph = apply_labels(self.references[row.record_id].graph, row.parsed_requirements())
            return reply(calls=[(NEW_PLAN, {"plan": to_compact(graph)})])
        if row.behavior == "edit":
            e = row.expect
            if e["edit"] == "link":
                args = {"room_a": e["a"], "room_b": e["b"], "kind": e["link"]}
                return reply(calls=[("set_link_kind", args)])
            if not made:
                new = {"type": e["type"], "x": 500, "y": 500, "size": 40}
                return reply(calls=[("add_room", new)])
            added = re.search(r"room (r\d+)", messages[-1]["content"])[1]
            link = {"room_a": added, "room_b": e["near"], "kind": "wall"}
            return reply(calls=[("add_link", link)])
        words = [group[0] for group in row.expect["groups"]]
        return reply(f"Do you mean {words[0]} or {words[-1]}?")


class LoopingClient:
    """Calls a harmless edit forever, or fails on a chosen request."""

    def __init__(self, fail_on=None):
        self.requests, self.fail_on = [], fail_on

    def complete(self, request):
        self.requests.append(request)
        if self.fail_on is not None and len(self.requests) == self.fail_on:
            raise ApiError("server restarting")
        return reply(calls=[("set_link_kind", {"room_a": "r0", "room_b": "r1", "kind": "door"})])


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


def test_the_worked_example_is_a_call_the_environment_accepts():
    prompt = system_prompt("one-example")
    assert prompt.startswith(system_prompt("as-is"))
    plan = prompt.split("You call new_plan with plan = ", 1)[1].splitlines()[0]
    empty = example_plan().model_copy(deep=True)
    empty.floors[0].rooms, empty.floors[0].links = [], []
    assert call_tool(empty, NEW_PLAN, {"plan": plan}).applied
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
    assert behavior["edit_add"]["correct_rate"] == 1.0
    assert behavior["restart"]["correct_rate"] == 1.0
    assert summary["strict"]["all_met_prompts"] >= 1  # the restart prompts are scored as draws
    silent, _, _ = run(sets, "behavior_test", "silent", tmp_path)
    assert silent["behavior"]["edit_door"]["correct_rate"] == 0.0
    assert silent["behavior"]["contradiction"]["correct_rate"] == 0.0


def test_a_run_resumes_and_skips_prompts_with_a_complete_transcript(sets, tmp_path):
    _, run_dir, client = run(sets, "template_test", "gold", tmp_path, limit=5)
    assert len(client.requests) == 10  # new_plan, then the plain reply
    rows = read_rows(sets / "template_test.jsonl")
    config = T2gConfig(
        run_name="gold", prompts=str(sets / "template_test.jsonl"), model="m", limit=8
    )
    again = GoldClient(rows, read_references(sets))
    collect_outputs(config, again, run_dir)
    assert len(again.requests) == 6
    with pytest.raises(ValueError):
        collect_outputs(config.model_copy(update={"temperature": 0.1}), again, run_dir)


def test_a_failure_inside_the_loop_is_retried_from_the_start(sets, tmp_path):
    rows = read_rows(sets / "behavior_test.jsonl")
    row = next(r for r in rows if r.kind == "edit_door")
    prompts = tmp_path / "one.jsonl"
    prompts.write_text(row.model_dump_json() + "\n")
    config = T2gConfig(run_name="r", prompts=str(prompts), model="m", max_tool_calls=3)
    run_dir = tmp_path / "run"
    collect_outputs(config, LoopingClient(fail_on=2), run_dir)
    first = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]
    assert first == [{"prompt_id": row.id, "api_error": "server restarting"}]
    again = LoopingClient()
    collect_outputs(config, again, run_dir)
    assert len(again.requests) == 3
    collect_outputs(config, LoopingClient(), run_dir)
    lines = (run_dir / "outputs.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[-1])["hit_call_cap"]


def test_the_loop_runs_each_call_and_sends_back_the_real_tool_response(sets, tmp_path):
    summary, run_dir, client = run(sets, "behavior_test", "gold", tmp_path)
    rows = {r.id: r for r in read_rows(sets / "behavior_test.jsonl")}
    outputs = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]
    add = next(o for o in outputs if rows[o["prompt_id"]].kind == "edit_add")
    assert [c["name"] for c in add["calls"]] == ["add_room", "add_link"]
    assert [len(t["tool_calls"]) for t in add["turns"]] == [1, 1, 0]
    assert add["final_content"] == "Done." and not add["hit_call_cap"]
    row = rows[add["prompt_id"]]
    expected = call_tool(row.start_graph(), "add_room", json.loads(add["calls"][0]["arguments"]))
    last = next(
        r for r in reversed(client.requests) if r["messages"][1]["content"].endswith(row.text)
    )
    tool_messages = [m for m in last["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["content"] == expected.text
    assert tool_messages[0]["content"].startswith("OK: added")
    assert tool_messages[1]["content"].startswith("OK: linked")
    assistant = [m for m in last["messages"] if m["role"] == "assistant"]
    assert [m["tool_calls"][0]["id"] for m in assistant] == [
        m["tool_call_id"] for m in tool_messages
    ]
    assert summary["call_cap_rate"] == 0.0


def test_a_transcript_that_reaches_the_call_cap_is_a_failure(sets, tmp_path):
    rows = [r for r in read_rows(sets / "behavior_test.jsonl") if r.kind == "edit_door"][:2]
    prompts = tmp_path / "doors.jsonl"
    prompts.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    for name in ("references.jsonl", "reference_tables.json"):
        (tmp_path / name).write_text((sets / name).read_text())
    config = T2gConfig(run_name="r", prompts=str(prompts), model="m")
    run_dir = tmp_path / "run"
    client = LoopingClient()
    collect_outputs(config, client, run_dir)
    assert len(client.requests) == 2 * 8
    outputs = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]
    assert all(len(o["calls"]) == 8 and o["hit_call_cap"] for o in outputs)
    summary = score_run(run_dir, rows)
    assert summary["call_cap_rate"] == 1.0
    assert summary["behavior"]["edit_door"]["call_cap_rate"] == 1.0
    assert summary["behavior"]["edit_door"]["correct_rate"] == 0.0


def test_the_http_client_keeps_structured_tool_calls():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c7",
                            "function": {"name": "new_plan", "arguments": '{"plan": "{}"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    reply = parse_chat_response(body, 0.1)
    assert reply.tool_calls == [{"id": "c7", "name": "new_plan", "arguments": '{"plan": "{}"}'}]
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


def test_a_generic_question_does_not_name_the_issue(sets, tmp_path):
    from vlm_floorplan.evaluation.t2g_score_one import score_behavior

    rows = {r.kind: r for r in read_rows(sets / "behavior_test.jsonl")}
    unsupported = rows["unsupported"]
    vague = {
        "content": "Which floor plan layout would you like? Someone will check.",
        "tool_calls": [],
    }
    assert score_behavior(unsupported, vague)["asked_question"]
    assert not score_behavior(unsupported, vague)["correct"]
    contradiction = rows["contradiction"]
    assert not score_behavior(contradiction, vague)["correct"]


def test_rescoring_without_references_fails_loudly(sets, tmp_path):
    _, run_dir, _ = run(sets, "template_test", "gold", tmp_path, limit=3)
    rows = read_rows(sets / "template_test.jsonl")[:3]
    with pytest.raises(FileNotFoundError):
        score_run(run_dir, rows, tmp_path / "elsewhere")
    assert score_run(run_dir, rows, sets)["real_reference"]["n_plans"] == 3
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "reference_tables.json").write_text((sets / "reference_tables.json").read_text())
    with pytest.raises(FileNotFoundError):
        score_run(run_dir, rows, partial)
    lines = (sets / "references.jsonl").read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["id"] != rows[0].record_id]
    (partial / "references.jsonl").write_text("".join(line + "\n" for line in kept))
    with pytest.raises(FileNotFoundError):
        score_run(run_dir, rows, partial)
    only_refs = tmp_path / "only_refs"
    only_refs.mkdir()
    (only_refs / "references.jsonl").write_text((sets / "references.jsonl").read_text())
    with pytest.raises(FileNotFoundError):
        score_run(run_dir, rows, only_refs)


def test_a_restart_asks_for_what_the_current_plan_lacks(sets, tmp_path):
    restarts = [r for r in read_rows(sets / "behavior_test.jsonl") if r.behavior == "restart"]
    assert restarts
    for row in restarts:
        assert not all_met(row.start_graph(), row.parsed_requirements()), row.id
    kept, _, _ = run(sets, "behavior_test", "keep", tmp_path)
    assert kept["behavior"]["restart"]["correct_rate"] == 0.0


def test_flag_rates_survive_a_failed_first_row(sets, tmp_path):
    _, run_dir, _ = run(sets, "behavior_test", "gold", tmp_path)
    rows = read_rows(sets / "behavior_test.jsonl")
    first_edit = next(r for r in rows if r.kind == "edit_door")
    lines = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]
    for line in lines:
        if line["prompt_id"] == first_edit.id:
            line.clear()
            line.update({"prompt_id": first_edit.id, "api_error": "boom"})
    (run_dir / "outputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    summary = score_run(run_dir, rows)
    entry = summary["behavior"]["edit_door"]
    assert entry["edit_effect_ok_rate"] < 1.0 and "edit_tools_only_rate" in entry
