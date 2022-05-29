from src.models.predict_model import main
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
            "--path_to_predictions_storage",
            "data/predictions",
        ],
    )

    assert result.exit_code == 0
