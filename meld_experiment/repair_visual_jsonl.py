#!/usr/bin/env python3
"""Conservatively recover unambiguous visual JSON formatting failures without a GPU."""
import argparse
import collections
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new path')
    records = []
    ids = set()
    counts = collections.Counter()
    with args.input.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['sample_id'] in ids or row.get('modality') != 'video':
                raise ValueError('Unexpected or duplicate sample ID')
            ids.add(row['sample_id'])
            raw = row.get('raw_output', '').strip()
            if row.get('status') == 'error' and raw:
                obj = None
                method = None
                try:
                    obj = json.loads(raw)
                    method = 'original_json'
                except json.JSONDecodeError:
                    if raw.endswith('"') and not raw.endswith('}'):
                        try:
                            obj = json.loads(raw + '\n}')
                            method = 'appended_missing_object_brace'
                        except json.JSONDecodeError:
                            pass
                if isinstance(obj, dict) and set(obj) == {'target_status', 'description'}:
                    status, desc = obj['target_status'], obj['description']
                    if status == 'visible' and isinstance(desc, str) and desc.strip():
                        row['original_error'] = row.get('error')
                        row.pop('error', None)
                        row.update(status='ok', target_status_model=status,
                                   candidate_description=desc.strip(), description=desc.strip(),
                                   speaker_verified=False, repair_method=method)
                        counts['recovered_description'] += 1
                    elif status == 'visible' and desc == '':
                        row['original_error'] = row.get('error')
                        row.pop('error', None)
                        row.update(status='target_unavailable', target_status_model='visible_empty_description',
                                   candidate_description=None, description=None, speaker_verified=False,
                                   repair_method='empty_description_marked_unavailable')
                        counts['empty_description'] += 1
            counts['status_' + row['status']] += 1
            records.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'input': str(args.input), 'output': str(args.output), 'samples': len(records), 'counts': dict(counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
