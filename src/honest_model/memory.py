import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Calculate the exact deep memory usage of a DataFrame in megabytes."""
    # df.memory_usage(deep=True) returns bytes per column; sum() gives total bytes
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the memory usage of a DataFrame by downcasting numeric colums
    and categorizing object columns."""

    # Step 1: Get the initial memory usage of the DataFrame
    start_mem = get_memory_usage_mb(df)

    print(f"Initial memory usage of DataFrame: {start_mem:.2f} MB")

    # step 2: Downcast columns to more efficient types
    for col in df.columns:
        if is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer", errors="raise")
        elif is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float", errors="raise")
        elif is_string_dtype(df[col]):
            df[col] = _downcast_string_column(df[col])

    # Step 3: Get the final memory usage of the DataFrame
    end_mem = get_memory_usage_mb(df)
    print(f"Final memory usage of DataFrame: {end_mem:.2f} MB")
    print(f"Memory usage reduction by: {(start_mem - end_mem) / start_mem * 100:.2f} %")

    # Step 4: Return the reduced DataFrame and the final memory usage
    return df


def _downcast_string_column(col: pd.Series, threshold: float = 0.5) -> pd.Series:
    """A helper function to downcast string column to category if the number of unique
    values is less than the specified threshold (low cardinality)."""
    unique_values = col.nunique()
    total_rows = len(col)
    if unique_values / total_rows < threshold:
        return col.astype("category")
    else:
        return col
