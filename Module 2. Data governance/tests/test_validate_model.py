from src.models import validate_model
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        validate_model,
        "../data/processed/test.csv ../models/finalized_model.pkl ../reports/metrics",
    )
    assert result.exit_code == 0
