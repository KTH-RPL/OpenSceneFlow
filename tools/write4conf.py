'''
# Usage for old model file to new version code
'''
import torch
import fire, time

def main(
    # download from: https://zenodo.org/records/12632962
    model_path: str = "/home/kin/model_zoo/backup/seflowpp_lr2e4.ckpt", 
    # new output weight file
    output_path: str = "/home/kin/model_zoo/seflowpp.ckpt",
):
    model = torch.load(model_path)
    model_name = model['hyper_parameters']['cfg']['model']['name']
    old_path = model['hyper_parameters']['cfg']['model']['target']['_target_']
    new_path = old_path.replace(f"scripts.network.models.{model_name}", "src.models")
    model['hyper_parameters']['cfg']['model']['target']['_target_'] = new_path
    # model['hyper_parameters']['cfg']['num_frames']=3
    torch.save(model, output_path)

def readmodel(
    model_path: str = "/home/kin/model_zoo/seflowpp.ckpt"
):
    model = torch.load(model_path)
    print(model['hyper_parameters']['cfg'], model['epoch'])

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(main)
    fire.Fire(readmodel)
    print(f"Time used: {time.time() - start_time:.2f} s")