import logging
from pathlib import Path

logger = logging.getLogger("yellow_taxi")


def setup_file_logging(path):
    """Send all yellow_taxi log messages to `path` (overwritten) and nowhere else."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return path


def log_schema(df, step):
    """Log df's schema after `step`. Driver-side only; does not trigger a Spark job."""
    logger.info("--- schema after %s (%d cols) ---\n%s", step, len(df.columns), df.schema.treeString())
