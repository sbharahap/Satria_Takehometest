import json

from scripts.compare_local_models import assess
from src.config import PROJECT_ROOT
from src.services.chat import UNKNOWN, UNVERIFIED


def cases():
    return json.loads((PROJECT_ROOT / "eval/local_model_cases.json").read_text())["cases"]


def test_comparison_requires_expected_source_location_not_just_correct_keywords():
    case = next(c for c in cases() if c["id"] == "hr03")
    answer = {
        "content": "Sekali dalam tiga tahun. [S1]",
        "citations": [{"filename": "people_policy_ptvi_pol_0014_id.pdf", "page": 3}],
    }
    assert assess(answer, case)["pass"]
    answer["citations"][0]["filename"] = "diversity_inclusion_ptvi_pol_0036_id.pdf"
    assert not assess(answer, case)["pass"]


def test_comparison_unknown_is_strict_and_does_not_reward_unverified_answer():
    case = {"unknown": True}
    assert assess({"content": UNKNOWN, "citations": []}, case)["pass"]
    assert not assess({"content": UNVERIFIED, "citations": []}, case)["pass"]
    assert not assess({"content": UNKNOWN, "citations": [{"page": 1}]}, case)["pass"]


def test_conditional_gift_answer_is_not_penalized_for_negated_blanket_ban():
    case = next(c for c in cases() if c["id"] == "legal04")
    answer = {
        "content": "Tidak semua hadiah dilarang; hadiah diperbolehkan sesuai batasan, prosedur, "
        "dan hukum yang berlaku. [S1]",
        "citations": [{"filename": "anti_corruption_ptvi_pol_0016_id.pdf", "page": 2}],
    }
    assert assess(answer, case)["pass"]
    answer["content"] = "Semua hadiah dilarang. [S1]"
    assert not assess(answer, case)["pass"]


def test_comparison_has_ten_unique_questions_per_agent():
    dataset = cases()
    assert len({c["id"] for c in dataset}) == len(dataset) == 20
    assert all(sum(c["agent"] == group for c in dataset) == 10 for group in ("hr", "legal"))
