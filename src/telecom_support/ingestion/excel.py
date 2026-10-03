"""Validate tickets and isolate held-out answers."""
import hashlib
from pathlib import Path
import pandas as pd
from .schemas import TicketRecord

REQUIRED = ["ticket_id", "created_at", "category", "product", "severity", "sentiment", "customer_complaint", "conversation", "resolution_steps", "resolution_summary", "split"]

def prepare_excel(path: Path, source_file: str, taxonomy: dict):
    frame = pd.read_excel(path, sheet_name="Tickets").dropna(how="all")
    if missing := set(REQUIRED) - set(frame.columns):
        raise ValueError(f"Missing ticket columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No tickets found.")
    for column in REQUIRED:
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Missing required values in {column}.")
        frame[column] = values
    if frame.ticket_id.duplicated().any():
        raise ValueError("Duplicate ticket IDs.")
    normalized = frame.customer_complaint.str.lower().str.replace(r"\s+", " ", regex=True)
    if normalized.duplicated().any():
        raise ValueError("Repeated normalized complaints; review source.")
    for column, registry in [("category", "categories"), ("product", "products"),
                             ("severity", "severities"), ("sentiment", "sentiments")]:
        if not frame[column].isin(taxonomy[registry]).all():
            raise ValueError(f"Unregistered {column}; review taxonomy.")
    if frame.loc[frame.category.eq("Other / Unknown"), "split"].ne("test").any():
        raise ValueError("Other / Unknown tickets must be test-only.")
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    train, test = [], []
    for row in frame[REQUIRED].to_dict("records"):
        record = TicketRecord(source_id=row["ticket_id"], source_file=source_file,
                              source_version=version, taxonomy_version=taxonomy["version"],
                              search_text=row["customer_complaint"], **row)
        (train if record.split == "train" else test).append(record)
    return train, test
