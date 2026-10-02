from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

SOURCE_FILES = [
    "conversations_source_01.xlsx",
    "conversations_source_02.xlsx",
]

EXPECTED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "secondary_category",
    "product",
    "severity",
    "sentiment",
    "customer_complaint",
    "conversation",
    "resolution_steps",
    "resolution_summary",
    "status",
    "split",
]


def main():
    dataframes = []
    source_records = []

    for filename in SOURCE_FILES:
        file_path = RAW_FOLDER / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {file_path}"
            )

        df = pd.read_excel(file_path, sheet_name="Tickets")

        # Ignore completely empty spreadsheet rows.
        df = df.dropna(how="all").reset_index(drop=True)

        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{filename} does not have the expected columns."
            )

        if len(df) != 100:
            raise ValueError(
                f"{filename}: expected 100 tickets, found {len(df)}."
            )

        for original_id in df["ticket_id"]:
            source_records.append({
                "source_file": filename,
                "original_ticket_id": original_id,
            })

        dataframes.append(df)
        print(f"Loaded {filename}: {len(df)} tickets")

    combined = pd.concat(dataframes, ignore_index=True)

    # Check complaints before changing IDs or saving anything.
    complaints = combined["customer_complaint"].astype("string")

    if complaints.isna().any() or complaints.str.strip().eq("").any():
        raise ValueError("A ticket has an empty customer complaint.")

    normalized = (
        complaints
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    duplicate_count = int(normalized.duplicated().sum())

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} repeated complaints. "
            "Review them before combining."
        )

    if not combined["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected train/test split value.")

    other_rows = combined["category"].eq("Other / Unknown")

    if combined.loc[other_rows, "split"].ne("test").any():
        raise ValueError("Other / Unknown tickets must be test-only.")

    # Both source files reuse IDs, so create fresh combined IDs.
    new_ids = [
        f"T-{number:04d}"
        for number in range(1, len(combined) + 1)
    ]

    combined["ticket_id"] = new_ids

    mapping = pd.DataFrame(source_records)
    mapping.insert(0, "ticket_id", new_ids)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_FOLDER / "conversations.xlsx"
    mapping_path = OUTPUT_FOLDER / "ticket_id_mapping.csv"

    combined.to_excel(
        output_path,
        sheet_name="Tickets",
        index=False,
    )

    mapping.to_csv(
        mapping_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nTotal tickets: {len(combined)}")
    print(f"Repeated complaints: {duplicate_count}")
    print(f"Unique ticket IDs: {combined['ticket_id'].nunique()}")

    print("\nTrain/test counts:")
    print(combined["split"].value_counts().to_string())

    print(f"\nCombined Excel saved to: {output_path}")
    print(f"ID mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()