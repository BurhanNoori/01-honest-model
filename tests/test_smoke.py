# step 1: Import the project package to verify pythonpath and packaging work
import honest_model


# step 2: Define a smoke test to verify test harness is functional
def test_package_importable() -> None:
    assert honest_model is not None
