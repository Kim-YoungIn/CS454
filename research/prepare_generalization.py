"""Verify pinned official fixtures and prepare RGB inputs without changing util.py.

Sources/splits are fixed in generalization_v1/protocol.json before any attack.
Network fetching is optional; verified originals are retained for offline reuse.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('research/generalization_v1'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((args.root/'protocol.json').read_text())
    records = []
    for item in protocol['items']:
        source = args.root/'originals'/item['original']
        if not source.exists() and args.download:
            source.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(item['url'], timeout=45) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != item['expected_sha256']:
                raise ValueError(f"upstream checksum mismatch: {item['id']}")
            source.write_bytes(raw)
        if digest(source) != item['expected_sha256']:
            raise ValueError(f"original checksum mismatch: {source}")
        target = args.root/'images'/f"{item['id']}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            size, mode = image.size, image.mode
            rgb = image.convert('RGB')
            if target.exists():
                with Image.open(target) as previous:
                    if previous.mode != 'RGB' or previous.size != rgb.size or previous.tobytes() != rgb.tobytes():
                        raise ValueError(f"prepared image differs: {target}")
            else:
                rgb.save(target)
        records.append(dict(id=item['id'], split=item['split'], path=str(target),
                            original_sha256=digest(source), prepared_sha256=digest(target),
                            original_mode=mode, width=size[0], height=size[1], source_url=item['url']))
    manifest = args.root/'manifest.csv'
    if manifest.exists():
        with manifest.open() as stream:
            old = list(csv.DictReader(stream))
        if old != [{key: str(value) for key, value in row.items()} for row in records]:
            raise ValueError('existing manifest differs; preserve the fixed dataset')
    else:
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    print(f'Verified {len(records)} originals and RGB images; splits unchanged.')


if __name__ == '__main__':
    main()
