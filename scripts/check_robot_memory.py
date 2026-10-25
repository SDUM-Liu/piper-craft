"""Check the bounded operation-memory directory after maintenance."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_vla.operation_memory import check  # noqa: E402

if __name__ == '__main__':
    counts = check()
    print('operation memory:', sum(counts.values()), '/260 lines', counts)
