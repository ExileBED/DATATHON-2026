"""Rebuild the 50/50 blend from saved test probabilities and check the upload.

Does not retrain. Confirms:

    submission = 0.5 * PAY-types (0.40929) + 0.5 * nested TE (10x10)

    python reconstruct_submission.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OFFICIAL = HERE / "submission_blend_te_paytypes.csv"
ID = "client_id"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    official = pd.read_csv(OFFICIAL)
    pay = pd.read_csv(ART / "test_pay_types.csv")
    te = pd.read_csv(ART / "submission_push4_te.csv")
    merged = official.merge(pay, on=ID, suffixes=("_upload", "_pay")).merge(
        te, on=ID, suffixes=("", "_te")
    )
    recon = 0.5 * merged["default_probability_pay"] + 0.5 * merged["default_probability"]
    err = np.max(np.abs(recon.to_numpy() - merged["default_probability_upload"].to_numpy()))
    print(f"upload rows              : {len(official)}")
    print(f"upload SHA-256           : {sha256(OFFICIAL)}")
    print(f"max |upload - 50/50 mix| : {err:.3e}")
    print("reconstructed from artifacts/test_pay_types.csv + artifacts/submission_push4_te.csv")
    if err > 1e-12:
        raise SystemExit("blend does not match the uploaded file")
    print("OK: uploaded file is exactly the 50/50 mix of the two saved components.")


if __name__ == "__main__":
    main()
