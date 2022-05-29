from src.visualization.visualize import main
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        main,
        [
            "--path_to_raw_data",
            "data/raw/winequality-red.csv",
            "--path_to_save_figs",
            "reports/figures",
        ],
    )
    assert result.exit_code == 0
