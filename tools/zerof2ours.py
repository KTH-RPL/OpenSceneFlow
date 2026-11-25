import torch
from pathlib import Path
import numpy as np
import fire, time

def main(
    # download from: https://github.com/kylevedder/zeroflow_weights/tree/master/argo
    model_path: str = "/home/kin/nsfp_distilatation_3x_49_epochs.ckpt", 
    # download from: https://zenodo.org/records/12632962
    reference_path: str = "/home/kin/fastflow3d.ckpt", 
    # new output weight file
    output_path: str = "/home/kin/zeroflow3x.ckpt",
):
    model = torch.load(model_path, weights_only=False)
    reference = torch.load(reference_path, weights_only=False)

    ref_model_weight = reference['state_dict']
    real_model_weight = model['state_dict']
    for k in real_model_weight.keys():
        if k not in ref_model_weight.keys():
            print(f"Warning: {k} not in reference model, not same model.")
            exit(0)
    
    reference['state_dict'] = real_model_weight
    torch.save(reference, output_path)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")