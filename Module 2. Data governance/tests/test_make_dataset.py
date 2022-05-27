from src.data import make_dataset
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        make_dataset, "../data/raw/winequality-red.csv ../data/processed/"
    )
    assert result.exit_code == 0
