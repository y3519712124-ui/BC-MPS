"""Run the complete recorded operating-point grid and matched spatial controls."""
import argparse
import importlib.util
from pathlib import Path
from _common import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=REPO / 'reproduced')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('public_posthoc', REPO / 'code/fixed_budget_posthoc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = args.output_root.resolve()
    module.BASE = module.ROOT / 'results/fixed_budget_calibration_v2'
    module.OUT = module.BASE / 'diagnostics'
    # The archived diagnostic expects probability arrays next to its recorded
    # checkpoint. Public weights remain immutable under artifacts/.
    original_records = module.BASE / 'records.jsonl'
    import json
    entries = [json.loads(s) for s in original_records.read_text().splitlines()]
    for record in entries:
        folder = module.ROOT / 'runs/fixed_budget_calibration_v2' / f"muufl_split{record['split_seed']}_s{record['seed']}_n0.40_fixed30"
        source = Path(record['checkpoint'])
        local_checkpoint = folder / 'final.pt'
        if not local_checkpoint.exists():
            import shutil
            shutil.copy2(source, local_checkpoint)
        record['checkpoint'] = str(local_checkpoint)
    original_records.write_text(''.join(json.dumps(x)+'\n' for x in entries), encoding='utf-8')
    module.main()


if __name__ == '__main__':
    main()
