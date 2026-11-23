import json
import random

from vlm_floorplan.evaluation.client import ApiError, ChatResponse
from vlm_floorplan.requirements import Adjacent, RoomCount, requirements_to_json
from vlm_floorplan.text2graph.paraphrase import Teacher, parse_extraction, plain_requirements

REQS = [
    RoomCount(type="bedroom", op="eq", n=3),
    Adjacent(a="kitchen", b="living", link="opening"),
]


class ScriptedClient:
    """Answers paraphrase calls from `paraphrases` and extraction calls from `extractions`."""

    def __init__(self, paraphrases, extractions):
        self.paraphrases, self.extractions = list(paraphrases), list(extractions)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        prompt = request["messages"][0]["content"]
        queue = self.extractions if prompt.startswith("Extract") else self.paraphrases
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(item, "", "stop", 1, 1, 0.0)


GOOD = "We need three bedrooms, and the kitchen should be open to the living room."
GOOD_JSON = json.dumps(requirements_to_json(REQS))


def teacher(paraphrases, extractions):
    client = ScriptedClient(paraphrases, extractions)
    return Teacher(client, "m"), client


def test_a_paraphrase_that_passes_both_checks_is_accepted():
    t, client = teacher([GOOD], ["```json\n" + GOOD_JSON + "\n```"])
    result = t.checked_paraphrase("orig", REQS, random.Random(1), attempts=2)
    assert result.text == GOOD and result.attempts == 1
    assert client.requests[0]["chat_template_kwargs"] == {"thinking": False}


def test_a_paraphrase_failing_the_slot_check_is_retried_without_a_round_trip():
    t, client = teacher(
        ["Four bedrooms and an open kitchen next to the lounge.", GOOD], [GOOD_JSON]
    )
    result = t.checked_paraphrase("orig", REQS, random.Random(1), attempts=3)
    assert result.text == GOOD and result.attempts == 2
    assert "extra number 4" in result.reasons[0]
    assert len(client.extractions) == 0 and len(client.requests) == 3


def test_a_blind_extraction_that_differs_drops_the_paraphrase():
    wrong = json.dumps(requirements_to_json([RoomCount(type="bedroom", op="ge", n=3), REQS[1]]))
    t, _ = teacher([GOOD, GOOD], [wrong, "not json"])
    result = t.checked_paraphrase("orig", REQS, random.Random(1), attempts=2)
    assert result.text is None
    assert "differs" in result.reasons[0] and "not a requirement list" in result.reasons[1]


def test_symmetric_adjacency_is_compared_without_order():
    swapped = json.dumps(
        requirements_to_json([REQS[0], Adjacent(a="living", b="kitchen", link="opening")])
    )
    t, _ = teacher([GOOD], [swapped])
    assert t.checked_paraphrase("orig", REQS, random.Random(1), attempts=1).text == GOOD


def test_an_api_error_counts_as_a_failed_attempt():
    t, _ = teacher([ApiError("boom"), GOOD], [GOOD_JSON])
    result = t.checked_paraphrase("orig", REQS, random.Random(1), attempts=2)
    assert result.text == GOOD and "api error" in result.reasons[0]


def test_parse_extraction_rejects_what_is_not_a_requirement_list():
    assert parse_extraction("[]") == []
    assert parse_extraction("no list here") is None
    assert parse_extraction('[{"kind":"made_up"}]') is None


def test_plain_requirements_are_fixed_phrases():
    assert plain_requirements(REQS) == ["3 bedrooms", "a kitchen open to a living room"]
