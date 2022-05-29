from src.data.make_dataset import make_dataset
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        make_dataset,
        [
            "--input_filepath",
            "data/raw/winequality-red.csv",
            "--output_filepath",
            "data/processed/",
        ],
    )
    assert result.exit_code == 0
