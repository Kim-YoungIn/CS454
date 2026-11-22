"""Prepare a preregistered, clean-correct labelled bank without running attacks.

Run after isolated model benchmarks finish. Every pool is fixed before clean
eligibility queries; all exclusions remain recorded. No confidence cutoff is used.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from PIL import Image

ARCHIVE_SHA256='64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5'
LABEL_SHA256='a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc'
WNIDS=('n01440764','n02102040','n02979186','n03000684','n03028079','n03394916','n03417042','n03425413','n03445777','n03888257')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('/tmp/cs454-imagenette2-160.tgz'))
    parser.add_argument('--labels',type=Path,default=Path('/tmp/cs454-imagenet-class-index.json'))
    parser.add_argument('--output',type=Path,default=Path('research/generalization_imagenette_v2'))
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; preserve the frozen selection')
    if sha(args.archive)!=ARCHIVE_SHA256 or sha(args.labels)!=LABEL_SHA256:
        raise ValueError('official archive or class-index checksum mismatch')
    source_labels=json.loads(args.labels.read_text())
    label_map={wnid:dict(index=int(index),label=label) for index,(wnid,label) in source_labels.items() if wnid in WNIDS}
    if set(label_map)!=set(WNIDS):raise ValueError('class mapping incomplete')
    names=ast.literal_eval(Path('category_names.txt').read_text())
    for value in label_map.values():value['assignment_name']=names[value['index']]
    by_class={wnid:[] for wnid in WNIDS}
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            if member.isfile() and len(path.parts)==4 and path.parts[1]=='val' and path.parts[2] in by_class and path.suffix.lower() in ('.jpg','.jpeg'):
                by_class[path.parts[2]].append(member.name)
    if any(len(paths)<20 for paths in by_class.values()):raise ValueError('insufficient official val members')
    args.output.mkdir(parents=True)
    (args.output/'originals').mkdir();(args.output/'images').mkdir()
    seed='CS454-labelled-generalization-v2'
    pools=[]
    for wnid,paths in by_class.items():
        ordered=sorted(paths,key=lambda path:(hashlib.sha256((seed+'\0'+path).encode()).hexdigest(),path))
        for parity,role in enumerate(('development','confirmation')):
            for rank,member in enumerate(ordered[parity::2][:10]):
                identity=f'{role}_{wnid}_{rank:02d}'
                pools.append(dict(id=identity,split=role,wnid=wnid,rank=rank,archive_member=member,
                    original_path=str(args.output/'originals'/f'{identity}.JPEG'),
                    expected_class=label_map[wnid]['index'],expected_name=label_map[wnid]['assignment_name']))
    protocol=dict(version=2,stage='Data prepared; attack methods/seeds/budgets must be frozen separately before confirmation.',
        source_url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',archive_sha256=ARCHIVE_SHA256,
        source_docs='https://github.com/fastai/imagenette#the-datasets',class_index_sha256=LABEL_SHA256,
        source_split='Official Imagenette val; ImageNet-derived, not asserted unseen during VGG pretraining.',
        preparation='Decode JPEG to RGB PNG without resize; unchanged instructor preprocessing handles224px input.',
        pool_rule='Per class, sort SHA256(seed+NUL+archive path), tie by path; even ranks development, odd confirmation; first10 candidates each.',
        selection_seed=seed,eligibility='First clean argmax equal to official1000-class label, per role/class; no confidence threshold or attack filtering.',
        duplicates='Exact original-byte duplicates are assigned to the first deterministic pool entry; later duplicate candidates are excluded.',
        reporting='Retain every scanned failure/exclusion and failure-to-fill class. Selected100% clean accuracy is an eligibility condition, not classifier accuracy.',
        scope='Ten labelled classes only, low-resolution160px source, two distinct photographs/class; descriptive transfer evidence, not population guarantees.',
        freeze='Pools and selector source are written before any clean forward. Confirmation attack outcomes must remain unused until selection.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    write_csv(args.output/'pools_before_clean.csv',pools)
    shutil.copy2(args.labels,args.output/'imagenet_class_index.json')
    (args.output/'labels.json').write_text(json.dumps(label_map,indent=2)+'\n')
    (args.output/'selector_snapshot.py').write_bytes(Path(__file__).read_bytes())
    needed={row['archive_member']:row for row in pools}
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            if member.name in needed:
                row=needed[member.name]
                with archive.extractfile(member) as source:Path(row['original_path']).write_bytes(source.read())
                row['original_sha256']=sha(Path(row['original_path']))
    owners={}
    for row in pools:
        if 'original_sha256' not in row:raise ValueError('candidate missing from archive')
        row['duplicate_of']=owners.setdefault(row['original_sha256'],row['id'])
        if row['duplicate_of']==row['id']:row['duplicate_of']=''
    write_csv(args.output/'pools.csv',pools)
    # Heavy imports and clean forwards occur only after pools/selection are fixed.
    import torch
    from util import get_image_tensor_from,get_model_most_likely_for
    torch.set_num_threads(4)
    import importlib.metadata
    import sys
    checkpoint=Path(torch.hub.get_dir())/'checkpoints/vgg11-8a719046.pth'
    environment=dict(python=sys.version,torch=torch.__version__,threads=torch.get_num_threads(),
        torchvision=importlib.metadata.version('torchvision'),pillow=importlib.metadata.version('Pillow'),
        source_sha256={name:sha(Path(name)) for name in ('util.py','grade.py','category_names.txt')},
        checkpoint_sha256=sha(checkpoint),selector_sha256=sha(Path(__file__)))
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    selected=[];scans=[];unfilled=[]
    with torch.inference_mode():
        for wnid in WNIDS:
            for role in ('development','confirmation'):
                found=False
                for row in (r for r in pools if r['wnid']==wnid and r['split']==role):
                    scanned=dict(row)
                    if row['duplicate_of']:
                        scanned.update(status='excluded_duplicate',predicted_class='',probability='')
                        scans.append(scanned);continue
                    path=args.output/'images'/f"{row['id']}.png"
                    with Image.open(row['original_path']) as image:
                        mode,size=image.mode,image.size;image.convert('RGB').save(path)
                    probability,predicted=get_model_most_likely_for(get_image_tensor_from(str(path)))
                    correct=predicted==row['expected_class']
                    scanned.update(status='selected' if correct else 'clean_label_mismatch',predicted_class=predicted,probability=probability)
                    scans.append(scanned)
                    if correct:
                        selected.append(dict(id=row['id'],split=role,path=str(path),wnid=wnid,
                            expected_class=row['expected_class'],expected_name=row['expected_name'],
                            original_sha256=row['original_sha256'],prepared_sha256=sha(path),
                            archive_member=row['archive_member'],source_rank=row['rank'],
                            source_mode=mode,source_width=size[0],source_height=size[1],
                            clean_class=predicted,clean_probability=probability))
                        found=True;break
                if not found:unfilled.append(dict(wnid=wnid,split=role))
    write_csv(args.output/'eligibility.csv',scans)
    if selected:write_csv(args.output/'manifest.csv',selected)
    (args.output/'unfilled.json').write_text(json.dumps(unfilled,indent=2)+'\n')
    (args.output/'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.relative_to(args.output)}\n' for path in sorted(args.output.rglob('*')) if path.is_file() and path.name!='SHA256SUMS'))
    print(f'Prepared {len(selected)} labelled clean-correct images; {len(scans)} scanned records, {len(unfilled)} unfilled role/classes. No attacks run.')
    return bool(unfilled)


if __name__=='__main__':raise SystemExit(main())
