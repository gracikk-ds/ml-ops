from src.models import train_model
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        train_model, "../data/processed/train.csv ../models/ ../reports/metrics"
    )
    assert result.exit_code == 0
