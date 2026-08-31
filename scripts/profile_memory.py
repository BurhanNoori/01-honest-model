"""
1. Loads each Home Credit CSV using pd.read_csv().
 2. Measures and prints baseline memory for each table.
 3. Calls reduce_memory_usage() on each.
 4. Prints before/after per-table and total working set size.

"""

import os

import pandas as pd

from honest_model.config import load_config
from honest_model.memory import reduce_memory_usage


def load_and_optimize_csvs() -> list[pd.DataFrame]:

    # Step 1: Only load the tables we actually need for modelling
    DATA_TABLES = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
    ]
    dfs = []
    raw_data_path = load_config().paths.raw_data_dir
    for file in os.listdir(raw_data_path):
        if file in DATA_TABLES:
            file_path = os.path.join(raw_data_path, file)
            print(f"Reading file: {file}...")

            df = pd.read_csv(file_path)

            # Step 2 & 3: Find baseline memory, Reduce memory usage
            # and print the reduction of the table
            reduce_memory_usage(df)
            print()
            # Step 4: Adding the optimized df into the list
            dfs.append(df)

    return dfs


# Usage
optimized_dfs = load_and_optimize_csvs()

total_memory = 0
for df in optimized_dfs:
    mem = df.memory_usage(deep=True).sum()
    total_memory += mem

print(
    f"Total memory occupied by all DataFrames: "
    f"{total_memory / (1024 * 1024 * 1024):.2f} GB"
)
