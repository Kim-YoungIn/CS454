"""Recheck online outputs at one thread and match regenerated parent values."""
import csv
import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
from util import get_image_tensor_from,get_model_predictions_for,get_L0_norm_for


@torch.inference_mode()
def main():
    output=Path(__file__).parent
    records=[]
    for row in csv.DictReader((output/'results.csv').open()):
        name=Path(row['image']).stem
        metadata=json.loads(row['engine_metadata'])
        parent_origin=metadata['chunks'][-1]['origin']
        parents=[]
        for seed,folder in ((454,'session2-production-five454'),(455,'session2-repeat-five455')):
            image=torch.load(Path('runs')/folder/f'raw_{name}_l0.png',weights_only=True)
            digest=hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()
            assert digest==parent_origin['parent_tensor_sha256'][str(seed)]
            parents.append(image)
        original=get_image_tensor_from(row['image']).contiguous()
        reference=int(get_model_predictions_for(original).argmax())
        result=torch.load(output/f'raw_{name}_l0.png',weights_only=True)
        predicted=int(get_model_predictions_for(result).argmax())
        distance=get_L0_norm_for(result-original)
        assert reference==int(row['reference_class']) and predicted!=reference
        assert distance==int(row['distance'])
        diagnostic=torch.load(Path(f'runs/session3-elite-{name}-ga/best.pt'),weights_only=True)
        strong,weak=sorted(parents,key=lambda parent:int(parent.ne(original).sum()))
        active=result.ne(original)
        record=dict(image=name,distance=distance,reference=reference,predicted=predicted,
                    threads=1,parent_value_hashes_match=True,
                    exact_tensor_matches_diagnostic=torch.equal(result,diagnostic),
                    strong_parent_loci=int((active & strong.ne(original)).sum()),
                    weak_parent_loci=int((active & weak.ne(original)).sum()))
        records.append(record)
    (output/'independent_validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
