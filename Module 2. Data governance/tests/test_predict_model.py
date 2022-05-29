from src.models import predict_model
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        predict_model,
        "../data/processed/test.csv ../models/finalized_model.pkl ../data/predictions",
    )
    assert result.exit_code == 0
