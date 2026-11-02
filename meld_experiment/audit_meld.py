"""Read-only MELD CSV/JSON checks; write the results to a separate report directory."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

LABELS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
# Official declare-lab/MELD contents API, checked 2026-10-26.
OFFICIAL = {
    'train': (9989, 'b91dbd634693608a8e77a00d9256ae9cbb51bc43'),
    'dev': (1109, '91dd39cd4a10a53eda3c8aaf1955d3eae795ae52'),
    'test': (2610, '357d4a5f7ec119215622d07ed55fa3cad5d8d4c3'),
}


def audit(root):
    result = {'data_root': str(root.resolve()), 'official_checked_on': '2026-10-26', 'splits': {}}
    for split, (expected_count, expected_sha) in OFFICIAL.items():
        path = root / f'{split}_sent_emo.csv'
        raw = path.read_bytes()
        blob_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        keys = [(int(r['Dialogue_ID']), int(r['Utterance_ID'])) for r in rows]
        expected = {
            key: {'speaker': r['Speaker'], 'text': r['Utterance'], 'emotion': r['Emotion']}
            for key, r in zip(keys, rows)
        }
        record = {
            'utterances': len(rows), 'expected_utterances': expected_count,
            'dialogues': len({key[0] for key in keys}),
            'emotion_counts': dict(Counter(r['Emotion'] for r in rows)),
            'duplicate_sample_keys': len(keys) - len(set(keys)),
            'empty_text': sum(not r['Utterance'].strip() for r in rows),
            'empty_speakers': sum(not r['Speaker'].strip() for r in rows),
            'invalid_labels': sorted({r['Emotion'] for r in rows} - set(LABELS)),
            'git_blob_sha1': blob_sha, 'official_git_blob_sha1': expected_sha,
            'identical_to_official': blob_sha == expected_sha,
            'official_url': f'https://github.com/declare-lab/MELD/blob/master/data/MELD/{split}_sent_emo.csv',
        }
        json_path = root / f'{split}.json'
        if json_path.exists():
            conversations = json.loads(json_path.read_text(encoding='utf-8'))
            actual, duplicate_keys, unsorted_dialogues = {}, 0, 0
            for dialogue in conversations:
                ids = [int(u['utterance_ID']) for u in dialogue['conversation']]
                unsorted_dialogues += ids != sorted(ids)
                for utterance in dialogue['conversation']:
                    key = (int(dialogue['conversation_ID']), int(utterance['utterance_ID']))
                    duplicate_keys += key in actual
                    actual[key] = {field: utterance[field] for field in ['speaker', 'text', 'emotion']}
            record['json_check'] = {
                'utterances': len(actual), 'duplicate_keys': duplicate_keys,
                'unsorted_dialogues': unsorted_dialogues,
                'missing_keys': len(expected.keys() - actual.keys()),
                'extra_keys': len(actual.keys() - expected.keys()),
                'mismatched_rows': sum(actual[k] != expected[k] for k in actual.keys() & expected.keys()),
                'matches_csv': actual == expected and duplicate_keys == 0 and unsorted_dialogues == 0,
            }
        else:
            record['json_check'] = None
        result['splits'][split] = record
    media_suffixes = {'.mp4', '.avi', '.mov', '.mkv', '.wav', '.flac', '.mp3', '.m4a'}
    result['media_inventory'] = dict(Counter(
        p.suffix.lower() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in media_suffixes
    ))
    result['media_scope'] = 'Only the supplied data directory and its subdirectories were inspected.'
    result['multimodal_alignment_verified'] = False
    return result


def write_report(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'meld_data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# MELD 数据核查报告', '', '核查日期：2026-10-26', '',
             f"检查目录：`{result['data_root']}`", '',
             '| 划分 | 话语数 | CSV 实际对话数 | 与官方文件一致 | JSON 与 CSV 逐条一致 |',
             '|---|---:|---:|---|---|']
    for split, r in result['splits'].items():
        j = r['json_check']
        lines.append(f"| {split} | {r['utterances']} | {r['dialogues']} | {'是' if r['identical_to_official'] else '否'} | {'是' if j and j['matches_csv'] else '未通过或未提供'} |")
    lines += ['', '三个 CSV 的 Git blob SHA-1 均与官方仓库 contents API 返回的文件标识核对；JSON 按对话和话语编号核对文本、说话人、情绪及顺序。', '',
              '| 情绪 | 训练 | 验证 | 测试 |', '|---|---:|---:|---:|']
    for label in LABELS:
        counts = [result['splits'][s]['emotion_counts'].get(label, 0) for s in OFFICIAL]
        lines.append(f"| {label} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += ['', '## 结论与下一步', '',
              '- CSV 逐项检查结果、空值、重复键及 JSON 对齐结果见同目录 JSON 报告。',
              '- 官方训练 CSV 实际有 1,038 个不同 Dialogue_ID；不要为了匹配简介中的 1,039 而增删样本。总话语数为 13,708。',
              f"- 当前检查目录内媒体文件统计：{result['media_inventory']}。未完成音视频可解码性及片段对齐验证。",
              '- 若媒体在其他目录，需将该位置纳入后续检查；当前结果不能说明整台电脑没有媒体。',
              '- 各 split 的 Dialogue_ID 会重复，样本唯一键必须包含 split。',
              '- 先准备纯文本分类数据（最多前 12 条 + 当前话语），保持官方 train/dev/test 划分；输入中不得放入真实情绪标签。',
              '- 多模态阶段需要官方话语级视频片段；可从视频音轨提取声音，不一定另行下载独立音频。',
              '- 在确认媒体完整前，不启动全量描述生成或 GRPO 训练。', '',
              '官方来源：https://github.com/declare-lab/MELD/tree/master/data/MELD', '']
    (output / 'MELD数据核查报告.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir)
    write_report(result, args.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
