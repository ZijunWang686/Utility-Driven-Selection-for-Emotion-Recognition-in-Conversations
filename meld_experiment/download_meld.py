"""Download the official MELD raw archive with resume, SHA-256 and safe extraction."""
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

URL = 'https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz?download=true'
SOURCES = {'huggingface': URL, 'umich': 'https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'}
EXPECTED_SIZE = 10878146150
SHA256 = 'a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message, flush=True)


def check_archive(path):
    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f'Archive size mismatch: {path.stat().st_size} != {EXPECTED_SIZE}')
    log('Checking SHA-256 of official archive...')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise ValueError('SHA-256 mismatch. File preserved for inspection; no extraction performed.')


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    log(f'Extracting {archive.name} into {root}')
    count = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive path: {member.name}')
            target = destination.joinpath(*relative.parts)
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe extraction target: {target}')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if shutil.disk_usage(destination).free < member.size + 1024**3:
                    raise RuntimeError('Not enough free disk space to continue extraction.')
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ValueError(f'Incomplete extracted file: {target}')
                count += 1
            else:
                raise ValueError(f'Links or special archive entries are not permitted: {member.name}')
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, required=True, help='A persistent data disk directory')
    p.add_argument('--extract', action='store_true')
    p.add_argument('--source', choices=tuple(SOURCES), default='huggingface')
    args = p.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.download.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another download/extraction is already running in this destination.')
    url = SOURCES[args.source]
    log(f'Download source: {url}')
    final, partial = root / 'MELD.Raw.tar.gz', root / 'MELD.Raw.tar.gz.part'
    if not final.exists():
        remaining = max(0, EXPECTED_SIZE - (partial.stat().st_size if partial.exists() else 0))
        reserve = 30 * 1024**3 if args.extract else 1024**3
        if shutil.disk_usage(root).free < remaining + reserve:
            raise SystemExit('Insufficient disk space: reserve download bytes plus 30 GiB for extraction (or omit --extract).')
        log(f'Downloading official archive ({EXPECTED_SIZE} bytes); resuming if a partial file exists.')
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '8', '--retry-delay', '10', '--connect-timeout', '30',
            '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-',
            '--output', str(partial), url], check=True)
        check_archive(partial)
        partial.rename(final)
    else:
        check_archive(final)
    manifest = {'url': url, 'archive_size': EXPECTED_SIZE, 'sha256': SHA256, 'sha256_verified': True,
                'official_metadata_checked_on': '2026-10-26', 'extracted': False}
    if args.extract:
        media = root / 'media'
        marker = root / 'outer_extraction.complete'
        if not marker.exists():
            safe_extract(final, media)
            marker.write_text(SHA256 + '\n')
        queue = sorted(media.rglob('*.tar.gz'))
        processed = set()
        while queue:
            nested = queue.pop(0)
            if nested in processed:
                continue
            processed.add(nested)
            nested_marker = nested.with_name(nested.name + '.extracted')
            target = nested.with_name(nested.name[:-7])
            if not nested_marker.exists():
                safe_extract(nested, target)
                nested_marker.write_text('complete\n')
            queue.extend(p for p in target.rglob('*.tar.gz') if p not in processed)
        videos = list(media.rglob('*.mp4'))
        manifest.update({'extracted': True, 'video_files': len(videos), 'media_root': str(media)})
        if not videos:
            raise RuntimeError('Archive extracted but no .mp4 clips found; inspect the archive layout.')
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log('DOWNLOAD VERIFIED' + (' AND EXTRACTION COMPLETE' if args.extract else ''))


if __name__ == '__main__':
    main()
