import pandas as pd

from honest_model.memory import get_memory_usage_mb, reduce_memory_usage


def test_reduce_memory_usage():
    # Create a sample DataFrame with various data types
    data = {
        "id": [1, 2, 3, 4, 5],
        "names": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "salaries": [
            500000000000000.0,
            600000000000000.0,
            70000000000000.0,
            800000000000000.0,
            900000000000000.0,
        ],
        "gender": ["F", "M", "M", "M", "F"],
    }

    df = pd.DataFrame(data)

    initial_memory = get_memory_usage_mb(df)

    final_memory = get_memory_usage_mb(reduce_memory_usage(df))

    assert final_memory < initial_memory, (
        "Memory usage should be reduced after downcasting and categorization."
    )

    assert df["id"].dtype == "int8", "ID column should be downcasted to int8."
    assert df["salaries"].dtype == "float64", (
        "Salaries column should be downcasted to float64."
    )
    assert df["names"].dtype != "category", (
        "Names column should not be converted to category."
    )
    assert df["gender"].dtype == "category", (
        "Gender column should be converted to category."
    )
