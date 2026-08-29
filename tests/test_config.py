import pathlib

import pytest

from honest_model.config import AppConfig, load_config


@pytest.fixture
def tmp_config_file():
    return pathlib.Path("config.yaml")


def test_config_loading_from_yaml(tmp_config_file):

    config = load_config(tmp_config_file)
    assert isinstance(config, AppConfig)
    assert config.paths.raw_data_dir == "data/raw"
    assert config.paths.processed_data_dir == "data/processed"
    assert config.paths.models_dir == "models"
    assert config.paths.reports_dir == "reports"
    assert config.data.target_col == "TARGET"
    assert config.data.id_col == "SK_ID_CURR"
    assert config.data.main_table == "application_train.csv"
    assert config.split.test_size == 0.2
    assert config.split.random_state == 42
    assert config.baseline_model.random_state == 42
    assert config.baseline_model.n_estimators == 100
    assert config.baseline_model.learning_rate == 0.05
    assert config.baseline_model.max_depth == 6
