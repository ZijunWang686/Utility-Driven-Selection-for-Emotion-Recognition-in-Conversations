#!/usr/bin/env python3
"""Make previously generated video candidates available for experiments, preserving provenance."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new filename')
    seen, counts = set(), {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding='utf-8') as src, args.output.open('x', encoding='utf-8') as dst:
        for line in src:
            record = json.loads(line)
            sample_id = record['sample_id']
            if sample_id in seen or record.get('modality') != 'video':
                raise ValueError(f'Invalid or duplicate video record: {sample_id}')
            seen.add(sample_id)
            if record.get('status') == 'needs_review':
                desc = record.get('candidate_description')
                if record.get('target_status_model') != 'visible' or not isinstance(desc, str) or not desc.strip():
                    raise ValueError(f'Invalid candidate: {sample_id}')
                record['status'] = 'ok'
                record['description'] = desc
                record['speaker_verified'] = False
                record['acceptance'] = 'model-generated; speaker not independently verified'
            counts[record['status']] = counts.get(record['status'], 0) + 1
            dst.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({'samples': len(seen), 'status_counts': counts, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
