from src.visualization import visualize
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        visualize, "../data/raw/winequality-red.csv ../reports/figures"
    )
    assert result.exit_code == 0
