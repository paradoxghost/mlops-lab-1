from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.food11.promote import GateError, promote_version


def run(accuracy, n="1000", dataset="mini", status="FINISHED"):
    return SimpleNamespace(info=SimpleNamespace(status=status), data=SimpleNamespace(
        metrics={"test_accuracy": accuracy}, params={"dataset": dataset, "evaluation_samples": n}))


@pytest.fixture
def client():
    result = Mock()
    result.get_model_version.return_value = SimpleNamespace(version="3", run_id="candidate")
    result.get_model_version_by_alias.return_value = SimpleNamespace(version="2", run_id="champion")
    result.get_run.side_effect = lambda run_id: run(0.85 if run_id == "candidate" else 0.8)
    return result


def test_promotes_qualifying_candidate(client):
    report = promote_version(client, "3")
    assert report["se"] == pytest.approx((0.8 * 0.2 / 1000) ** 0.5)
    assert report["threshold"] == pytest.approx(0.8252982212813471)
    assert report["decision"] == "promoted"
    client.set_registered_model_alias.assert_called_once_with("food11", "champion", "3")
    tags = {call.args[2]: call.args[3] for call in client.set_model_version_tag.call_args_list}
    assert tags["quality_gate.candidate_accuracy"] == "0.85"
    assert tags["quality_gate.champion_accuracy"] == "0.8"
    assert tags["quality_gate.decision"] == "promoted"


def test_nonqualifying_candidate_is_challenger(client):
    client.get_run.side_effect = lambda _: run(0.8)
    report = promote_version(client, "3")
    assert report["decision"] == "challenger"
    client.set_registered_model_alias.assert_called_once_with("food11", "challenger", "3")


def test_threshold_equality_qualifies(client):
    threshold = 0.8 + 2 * (0.8 * (1 - 0.8) / 1000) ** 0.5
    client.get_run.side_effect = lambda rid: run(threshold if rid == "candidate" else 0.8)
    assert promote_version(client, "3")["decision"] == "promoted"


def test_already_champion_is_noop(client):
    client.get_model_version.return_value = SimpleNamespace(version="2", run_id="champion")
    assert promote_version(client, "2")["decision"] == "already_champion"
    client.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("bad_run", [run(None), run(float("nan")), run(1.1), run(0.8, "0"),
                                    run(0.8, "unknown"), run(0.8, dataset="unknown"),
                                    run(0.8, status="FAILED")])
def test_bad_evidence_never_changes_alias(client, bad_run):
    client.get_run.return_value = bad_run
    client.get_run.side_effect = None
    with pytest.raises(GateError):
        promote_version(client, "3")
    client.set_registered_model_alias.assert_not_called()
    client.set_model_version_tag.assert_not_called()


@pytest.mark.parametrize("candidate_run", [run(0.9, "3347", "processed"), run(0.9, "1096")])
def test_incomparable_evaluation_sets(client, candidate_run):
    client.get_run.side_effect = lambda rid: candidate_run if rid == "candidate" else run(0.8)
    with pytest.raises(GateError, match="same evaluation"):
        promote_version(client, "3")
    client.set_registered_model_alias.assert_not_called()


def test_missing_alias_or_unreachable_server_never_promotes(client):
    client.get_model_version_by_alias.side_effect = ConnectionError("unavailable")
    with pytest.raises(ConnectionError):
        promote_version(client, "3")
    client.set_registered_model_alias.assert_not_called()


def test_invalid_version(client):
    with pytest.raises(GateError):
        promote_version(client, "-1")
    client.get_model_version.assert_not_called()


def test_concurrent_champion_change_aborts(client):
    client.get_model_version_by_alias.side_effect = [SimpleNamespace(version="2", run_id="champion"),
                                                   SimpleNamespace(version="4", run_id="another")]
    with pytest.raises(GateError, match="changed"):
        promote_version(client, "3")
    client.set_registered_model_alias.assert_not_called()
