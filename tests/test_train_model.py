from src.models.train_model import main
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        main,
        [
            "--path_to_dataset",
            "data/processed/train.csv",
            "--experiment_name",
            "test_run_experiment" "--registered_model_name",
            "test_run_model",
        ],
    )
    assert result.exit_code == 0
