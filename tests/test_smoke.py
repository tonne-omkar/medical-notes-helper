"""A quick check that the command line starts up. If this fails, something basic is broken."""


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out
