from pathlib import Path

from pipeline_v11 import BASE, run_pipeline


if __name__ == "__main__":
    run_pipeline("val", BASE / "submissions" / "val_pred_v11.csv", evaluate=True)
