from __future__ import annotations

import asyncio

from spice.sandbox.base import ExecResult
from spice.tools.verification import run_verified_command, verification_scope, verification_summary, workspace_fingerprint


class Environment:
    name = "test"

    def __init__(self, action=None, exit_code=0):
        self.action = action
        self.exit_code = exit_code

    async def run(self, command, **kwargs):
        if self.action:
            self.action()
        return ExecResult("output", "output", "", self.exit_code)


def run_check(tmp_path, environment):
    records = []

    async def run():
        with verification_scope(records.append):
            await run_verified_command("test-command", root=tmp_path, cwd=tmp_path, environment=environment, timeout=5)

    asyncio.run(run())
    return records


def test_evidence_is_bound_to_code_and_environment(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    records = run_check(tmp_path, Environment())
    assert verification_summary(records, root=tmp_path, environment="test")["status"] == "verified"
    assert verification_summary(records, root=tmp_path, environment="docker")["status"] == "stale"
    source.write_text("changed")
    assert verification_summary(records, root=tmp_path, environment="test")["status"] == "stale"


def test_rerunning_in_current_environment_supersedes_old_evidence(tmp_path):
    records = run_check(tmp_path, Environment())
    new_environment = Environment()
    new_environment.name = "docker"
    records.extend(run_check(tmp_path, new_environment))
    summary = verification_summary(records, root=tmp_path, environment="docker")
    assert summary["status"] == "verified"
    assert len(summary["records"]) == 1


def test_code_changes_during_verification_are_not_verified(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("before")
    records = run_check(tmp_path, Environment(lambda: source.write_text("after")))
    assert records[0]["status"] == "stale"


def test_failed_checks_and_other_task_records_cannot_prove_completion(tmp_path):
    records = run_check(tmp_path, Environment(exit_code=1))
    assert verification_summary(records, root=tmp_path)["status"] == "failed"
    assert verification_summary(records, root=tmp_path, task_id="different-task")["status"] == "unverified"


def test_malformed_evidence_does_not_default_to_verified(tmp_path):
    records = [{"command": "test", "cwd": str(tmp_path), "status": "invented"}]
    assert verification_summary(records, root=tmp_path)["status"] == "unverified"


def test_incomplete_fingerprint_cannot_prove_completion(tmp_path):
    fingerprint = workspace_fingerprint(tmp_path)
    records = [{"command": "test", "cwd": str(tmp_path), "status": "verified", "after": {**fingerprint, "complete": False}}]
    assert verification_summary(records, root=tmp_path)["status"] == "stale"


def test_old_failure_becomes_pending_reverification_when_code_changes(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("old code")
    records = run_check(tmp_path, Environment(exit_code=1))
    path.write_text("new code")

    summary = verification_summary(records, root=tmp_path, environment="test")

    assert summary["status"] == "stale"
    assert summary["records"][0]["recorded_status"] == "failed"
    assert records[0]["status"] == "failed"


def test_current_failure_still_blocks_unrelated_success(tmp_path):
    records = run_check(tmp_path, Environment(exit_code=1))
    successes = run_check(tmp_path, Environment())
    successes[0]["command"] = "different-check"
    assert verification_summary([*records, *successes], root=tmp_path)["status"] == "failed"


def test_rerunning_failed_check_can_establish_current_success(tmp_path):
    records = run_check(tmp_path, Environment(exit_code=1))
    records.extend(run_check(tmp_path, Environment()))
    assert verification_summary(records, root=tmp_path)["status"] == "verified"
