from convai.cli import DEMO_SCRIPT, main, run_demo


def test_cli_single_message(capsys):
    assert main(["--message", "Where is my order ORD-123456?"]) == 0
    out = capsys.readouterr().out
    assert "order_status" in out
    assert "ORD-123456" in out


def test_cli_eval_passes(capsys):
    assert main(["--eval"]) == 0
    out = capsys.readouterr().out
    assert "Intent accuracy: 1.000" in out
    assert "Final-action accuracy: 1.000" in out


def test_cli_demo_runs(capsys):
    run_demo()
    out = capsys.readouterr().out
    assert "Assistant" in out
    assert "Analytics:" in out
    assert "refund" in out.lower()


def test_demo_script_covers_key_flows():
    joined = " ".join(DEMO_SCRIPT).lower()
    assert "return" in joined
    assert "refund" in joined
    assert "goodbye" in joined
