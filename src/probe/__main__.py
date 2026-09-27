"""Run the held-out probe: python -m src.probe"""

from src.config import PROBE_REPORT_PATH
from src.probe.probes import run_probe


def main() -> None:
    report = run_probe()
    print(f"{PROBE_REPORT_PATH} lift={report['prior_genetics_auroc_lift']}")


if __name__ == "__main__":
    main()
