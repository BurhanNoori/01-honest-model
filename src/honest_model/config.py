# step 1: Import necessary libraries
from pathlib import Path

import yaml  # yaml to read and deserialize yaml into python objects
from pydantic import BaseModel, Field  # lib for data validation and settings management


# Step 2: Define configuration classes for different aspects of the project
#  and wrap them in a main config say AppConfig
class PathConfig(BaseModel):
    """
    Configuration for file paths used in the project."""

    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    models_dir: str = Field(default="models")
    reports_dir: str = Field(default="reports")


class DataConfig(BaseModel):
    """
    Configuration for data-related settings."""

    target_col: str = Field(default="TARGET")
    id_col: str = Field(default="SK_ID_CURR")
    main_table: str = Field(default="application_train.csv")


class SplitConfig(BaseModel):
    """
    Configuration for data splitting settings."""

    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """
    Configuration for model-related settings."""

    random_state: int = Field(default=42)
    n_estimators: int = Field(default=100)
    learning_rate: float = Field(default=0.05)
    max_depth: int = Field(default=6)
    model_name: str = Field(default="honest_model")
    model_version: str = Field(default="1.0.0")


class AppConfig(BaseModel):
    """
    Main configuration class that aggregates all other configurations."""

    paths: PathConfig
    data: DataConfig
    split: SplitConfig
    baseline_model: ModelConfig


# Step 3: Define a function to load the configuration from a YAML file
# and validate it using the AppConfig class
def load_config(config_path: Path | str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        # Pydantic's model_validate method will validate the
        # loaded config against the AppConfig schema
        return AppConfig.model_validate(config_dict)
