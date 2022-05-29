from src.models.validate_model import main
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        main,
        [
            "--path_to_dataset",
            "data/processed/test.csv",
            "--path_to_model_pkl",
            "models/finalized_model.pkl",
            "--path_to_metrics_storage",
            "reports/metrics",
        ],
    )
    assert result.exit_code == 0
