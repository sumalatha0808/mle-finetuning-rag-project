"""Keep single-system commands lightweight and smoke runs separate."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from solution.hybrid_project import cli
from solution.hybrid_project.config import Settings
from solution.hybrid_project.generation import ExtractiveSmokeGenerator


@pytest.mark.parametrize(
    ("system", "adapters", "needs_retriever"),
    [
        ("base-only", [False], False),
        ("rag-base", [False], True),
        ("rag-lora", [True], True),
        (None, [False, True], True),
    ],
)
def test_only_requested_models_are_loaded(
    monkeypatch, system, adapters, needs_retriever
):
    generator = Mock(
        side_effect=lambda settings, use_adapter: ExtractiveSmokeGenerator(
            str(use_adapter)
        )
    )
    retriever = Mock()
    monkeypatch.setattr(cli, "LocalModelGenerator", generator)
    monkeypatch.setattr(cli, "_retriever", retriever)
    cli._pipeline(Settings.from_env(), False, system)
    assert [call.kwargs["use_adapter"] for call in generator.call_args_list] == adapters
    assert retriever.called == needs_retriever


@pytest.mark.parametrize("system", ["base-only", "rag-base"])
def test_baseline_cli_does_not_require_adapter(monkeypatch, smoke_pipeline, system):
    def generator(settings, use_adapter):
        assert not use_adapter
        return ExtractiveSmokeGenerator("baseline")

    monkeypatch.setattr(cli, "LocalModelGenerator", generator)
    monkeypatch.setattr(cli, "_retriever", lambda *args: smoke_pipeline.retriever)
    result = CliRunner().invoke(
        cli.main, ["ask", "--system", system, "Standard shipping time?"]
    )
    assert result.exit_code == 0, result.output


def test_smoke_training_is_rejected_before_loading(monkeypatch):
    train = Mock()
    monkeypatch.setattr(cli, "train_adapter", train)
    result = CliRunner().invoke(cli.main, ["--hash-smoke", "train"])
    assert result.exit_code == 2
    assert "training is real" in result.output
    train.assert_not_called()


@pytest.mark.parametrize("smoke", [True, False])
def test_default_reports_are_separate(tmp_path, monkeypatch, smoke):
    settings = replace(Settings.from_env(), project_root=tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    monkeypatch.setattr(cli, "_pipeline", Mock())
    summary = SimpleNamespace(
        retrieval_recall_at_k=1.0,
        systems=[
            SimpleNamespace(system=s, answer_relevance=1.0)
            for s in ("rag-base", "rag-lora")
        ],
    )
    writer = Mock(return_value=summary)
    monkeypatch.setattr(cli, "write_report", writer)
    result = CliRunner().invoke(
        cli.main, (["--hash-smoke"] if smoke else []) + ["evaluate"]
    )
    assert result.exit_code == 0, result.output
    name = "smoke_evaluation.md" if smoke else "reference_evaluation.md"
    assert writer.call_args.args[1] == tmp_path / "solution" / "reports" / name


@pytest.mark.parametrize("suffix", ["md", "json"])
def test_smoke_cannot_overwrite_reference(tmp_path, monkeypatch, suffix):
    settings = replace(Settings.from_env(), project_root=tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    pipeline = Mock()
    monkeypatch.setattr(cli, "_pipeline", pipeline)
    target = tmp_path / "solution" / "reports" / f"reference_evaluation.{suffix}"
    target.parent.mkdir(parents=True)
    target.write_text("real benchmark")
    result = CliRunner().invoke(
        cli.main, ["--hash-smoke", "evaluate", "--output", str(target)]
    )
    assert result.exit_code == 2
    assert target.read_text() == "real benchmark"
    pipeline.assert_not_called()
