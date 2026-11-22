"""Independently recheck successful experiment tensors using one CPU thread."""
import argparse
import csv
import math
from pathlib import Path

import torch
from util import get_image_tensor_from, get_model_most_likely_for
from verify_artifacts import METRICS, read_tensor


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; preserve previous verification')
    torch.set_num_threads(1)
    references={};records=[]
    for directory in args.directories:
        with (directory/'results.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            if row['status']!='pass':
                records.append(dict(experiment=str(directory),image=Path(row['image']).stem,
                    norm=row['norm'],method=row.get('method',''),reported_distance=row.get('distance',''),
                    measured_distance='',reference_class=row['reference_class'],predicted_class='',
                    same_predicted_class='',status='source_not_successful'))
                continue
            key=row['image']
            if key not in references:
                original=get_image_tensor_from(key).contiguous()
                _,reference=get_model_most_likely_for(original)
                references[key]=(original,reference)
            original,reference=references[key]
            folder=Path(row['configuration']).parent
            image=read_tensor(folder/f"raw_{Path(key).stem}_{row['norm']}.png",original)
            measured=METRICS[row['norm']](image-original)
            _,predicted=get_model_most_likely_for(image)
            reported=float(row['distance'])
            close=(measured==reported if row['norm']!='l2' else
                   math.isclose(measured,reported,rel_tol=1e-6,abs_tol=1e-7))
            status='pass' if close and reference==int(row['reference_class']) and predicted!=reference else 'fail'
            records.append(dict(experiment=str(directory),image=Path(key).stem,norm=row['norm'],method=row['method'],
                reported_distance=reported,measured_distance=measured,reference_class=reference,
                predicted_class=predicted,same_predicted_class=predicted==int(row['predicted_class']),status=status))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    failures=sum(row['status']=='fail' for row in records)
    print(f'One-thread verification: {sum(row["status"]=="pass" for row in records)} successful artifacts pass; {failures} failures; '+
          f'{sum(row["status"]=="source_not_successful" for row in records)} original unsuccessful trials retained.')
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
