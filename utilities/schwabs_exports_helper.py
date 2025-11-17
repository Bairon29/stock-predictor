import os
import pandas as pd
from data_helpers import (
    load_data,
    save_data,
    merge_data,
    strip_whitespace_columns,
    strip_whitespace_columns,
    reset_dataframe_index,
    set_dataframe_index,
)


if __name__ == "__main__":
    df = load_data(
        symbol="2025-10-03-WatchListScanner-copy",
        folder="/Volumes/Bairon/ModalTrain/Stock_scans_exports",
        has_multi_level_headers=True,
        index_col=0,  # Use first column as index if "Symbol" is not found
        skiprows=1,
    )
    
    # strip whitespace from column names
    df = strip_whitespace_columns(df)
    # make sure index is named 'Symbol' so reset_index produces that column name
    df = set_dataframe_index(df, 'Symbol')
    # move index into a column
    # df = reset_dataframe_index(df)

    # Load existing summary if exists to preserve data
    existing_summary_path = "/Volumes/Bairon/ModalTrain/DataTestWorkSpace"
    existing_df = load_data(
        symbol="Watchlist_summary222",
        folder=existing_summary_path,
        has_multi_level_headers=False,
        index_col=0,  # Use first column as index if "Symbol" is not found
        skiprows=0,
    )
    # strip whitespace from column names
    existing_df = strip_whitespace_columns(existing_df)
    # make sure index is named 'Symbol' so reset_index produces that column name
    existing_df = set_dataframe_index(existing_df, 'Symbol')
    # Print existing dataframe
    print("\n ============= Existing DataFrame ============= \n")
    print(existing_df)
    print("\n ============= End Existing DataFrame ============= \n")

    new_update_df = merge_data(existing_df, df, keep='first')
    
    # Print new_update_df dataframe
    print("\n ============= Updated DataFrame ============= \n")
    print(new_update_df)
    print("\n ============= End Updated DataFrame ============= \n")
    # add the four new columns (empty strings or NaN as you prefer)
    # new_cols = ["daily_start", "daily_end", "1min_start", "1min_end"]
    # for c in new_cols:
    #     if c not in df.columns:
    #         df[c] = ""

    # select and save the desired columns
    # out = df[["Symbol", "Description"] + new_cols]
    # out_filepath = os.path.join("/Volumes/Bairon/ModalTrain/Stock_scans_exports", "Watchlist_summary.csv")
    # out.to_csv(out_filepath, index=False)
    # print("Saved:", out_filepath)
    # move index into a column
    new_update_df = reset_dataframe_index(new_update_df)
    save_data(new_update_df[["Symbol", "Description", "daily_start", "daily_end", "1min_start", "1min_end"]], "Watchlist_summary333", "/Volumes/Bairon/ModalTrain/DataTestWorkSpace", index=False)
    # print(out)

    # print a specific column
    # print(df)
