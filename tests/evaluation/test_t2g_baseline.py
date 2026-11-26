import json

from test_t2g_harness import GoldClient, sets  # noqa: F401  (the module-scoped set fixture)

from vlm_floorplan.evaluation.baseline_job import (
    run_text_baseline,
    serve_env,
    spec_from_parameters,
)
from vlm_floorplan.evaluation.baseline_spec import BaselineSpec, pack_sets, unpack_sets
from vlm_floorplan.evaluation.collect import _check_result
from vlm_floorplan.evaluation.run_dir import write_manifest
from vlm_floorplan.evaluation.t2g_cli import headline
from vlm_floorplan.text2graph.rows import read_references, read_rows


def test_the_text_task_round_trips_through_the_job_parameters():
    spec = BaselineSpec(run_name="r", task="text2graph", variants=["as-is"], prompt_sets=["a", "b"])
    again = spec_from_parameters(spec.job_parameters())
    assert (again.task, again.variants, again.prompt_sets) == ("text2graph", ["as-is"], ["a", "b"])
    old = spec.job_parameters()
    for key in ("task", "variants", "prompt_sets"):
        del old[key]
    assert spec_from_parameters(old).task == "image"  # a launch from before the text task


def test_the_set_directory_packs_and_unpacks(sets, tmp_path):  # noqa: F811
    archive = tmp_path / "sets.tar.gz"
    assert pack_sets(sets, archive) >= 7
    unpacked = unpack_sets(archive, tmp_path / "out")
    assert (unpacked / "template_test.jsonl").read_text() == (
        sets / "template_test.jsonl"
    ).read_text()
    assert (unpacked / "references.jsonl").exists() and (
        unpacked / "reference_tables.json"
    ).exists()


def test_the_job_runs_every_variant_and_set_and_leaves_a_verifiable_result(sets, tmp_path):  # noqa: F811
    out = tmp_path / "run"
    out.mkdir()
    rows = read_rows(sets / "template_test.jsonl") + read_rows(sets / "behavior_test.jsonl")
    client = GoldClient(rows, read_references(sets))
    spec = BaselineSpec(
        run_name="r",
        task="text2graph",
        prompt_sets=["template_test", "paraphrase_test", "behavior_test", "handwritten_test"],
        limit=6,
        concurrency=2,
    )
    result = run_text_baseline(spec, sets, out, client, "http://unused/v1")
    # The paraphrase and hand-written sets are empty here and are skipped, not failed.
    assert result["errors"] == {}
    assert sorted(result["runs"]) == [
        "as-is-behavior-test",
        "as-is-template-test",
        "one-example-behavior-test",
        "one-example-template-test",
    ]
    write_manifest(out)
    assert any("metrics.json" in c for c in _check_result(out))
    systems = {r["messages"][0]["content"] for r in client.requests}
    assert len(systems) == 2  # the as-is and the worked-example prompts


def test_the_headline_shows_the_numbers_to_read_first(sets, tmp_path):  # noqa: F811
    from test_t2g_harness import run

    summary, _, _ = run(sets, "template_test", "gold", tmp_path)
    text = headline(summary)
    assert "tool-call rate (new_plan): 1.000" in text and "all-met (strict): 1.000" in text
    assert "L4: 1.000" in text and "circulation_rate" in text
    json.dumps(summary, default=str)


def test_the_text_job_serves_a_context_that_holds_the_whole_agent_loop():
    # The loop's last request resends the first prompt (about 4K tokens) and seven earlier turns
    # of up to max_tokens each, then asks for max_tokens more: over 8 * 8192 tokens in all.
    text = BaselineSpec(run_name="r", task="text2graph", max_tokens=8192)
    context = int(serve_env(text, 8000)["MAX_MODEL_LEN"])
    assert 8 * 8192 + 4096 < context <= 262144  # Qwen3.5-4B's native context
    image = BaselineSpec(run_name="r", task="image", max_tokens=12288)
    assert serve_env(image, 8000)["MAX_MODEL_LEN"] == "20480"  # one request per plan
