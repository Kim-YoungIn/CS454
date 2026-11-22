"""Restore legacy/tmp aliases to ignored persistent workspace runtime caches."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for link,target in ((Path('/tmp/cs454-baseline-env'),root/'.runtime/python-env'),
                    (Path('/tmp/cs454-torch'),root/'.runtime/torch-cache')):
    if not target.exists():
        raise SystemExit(f'Missing persistent runtime cache: {target}; reinstall pinned requirements.')
    if link.is_symlink() and link.resolve()==target.resolve():continue
    if link.exists() or link.is_symlink():raise SystemExit(f'Preserve unexpected existing path: {link}')
    link.symlink_to(target,target_is_directory=True)
print('Runtime and checkpoint aliases are ready.')
