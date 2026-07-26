import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from Wayformer.wayformer import WayformerPL
from Wayformer.wayformer_config import batch_list_to_batch_tensors
from Wayformer.wf_dataset import NuplanDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    user_args = parser.parse_args()
    # res_path = Path(user_args.ckpt).parent.parent / 'eval.txt'

    res_path = Path('./eval_result.txt')
    print(f"Results will be saved to: {res_path.absolute()}")
    print(res_path)
    model: WayformerPL = WayformerPL.load_from_checkpoint(
        checkpoint_path=user_args.ckpt,
        weights_only=False
    )
    model.to('cuda:0')
    model.eval()

    args = model.config
    val_data = NuplanDataset(args, 'val',reuse_cache=True, val_ratio=0.1)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers//2,
        collate_fn=batch_list_to_batch_tensors
    )
    num_samples = 0
    ADEs = []
    FDEs = []
    missed = 0

    for i, batch in enumerate(tqdm(val_loader)):
        metrics = model.evaluate_model(batch)
        batch_size = len(batch)
        ADEs.append(metrics.ADE * batch_size)
        FDEs.append(metrics.FDE * batch_size)
        missed += metrics.missed
        num_samples += batch_size


    missRate = missed / num_samples
    ADE = sum(ADEs) / num_samples
    FDE = sum(FDEs) / num_samples
    with open(res_path, 'w') as f:
        f.write(f'ADE: {ADE}\nFDE: {FDE}\nmissRate: {missRate}\n')
