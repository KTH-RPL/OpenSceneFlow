Tools
---

Here we introduce some tools to help you:
- visualize the data and results.
- convert the pretrained model from others.
- ... More to come.

## Visualization

run `tools/visualization.py` to view the scene flow dataset with ground truth flow. Note the color wheel in under world coordinate.

```bash
# Visualize flow with color coding
python tools/visualization.py vis --data_dir /path/to/data --res_name flow

# Compare multiple results side-by-side
python tools/visualization.py vis --data_dir /path/to/data --res_name "[flow, deflow, deltaflow, ssf]"

# Show flow as vector lines
python tools/visualization.py vector --data_dir /path/to/data

# Check flow with pc0, pc1, and flowed pc0
python tools/visualization.py check --data_dir /path/to/data

# Show error heatmap
python tools/visualization.py error --data_dir /path/to/data --res_name "[flow, deflow, deltaflow, ssf]"
```

Demo Effect (press `SPACE` to stop and start in the visualization window):

https://github.com/user-attachments/assets/f031d1a2-2d2f-4947-a01f-834ed1c146e6

**Tips**: To quickly create qualitative results for all methods, you can use multiple results comparison mode, select a good viewpoint and then save screenshots for all frames by pressing `P` key. You will found all methods' results are saved in the output folder (default is `logs/imgs`).

## Quick Read .h5 Files

You can quickly read all keys and shapes in a .h5 file by:

```bash
python tools/read_h5.py --file_path /path/to/file.h5
```

## Conversion

run `tools/zero2ours.py` to convert the ZeroFlow pretrained model to our codebase. 

```bash
python tools/zero2ours.py --model_path /home/kin/nsfp_distilatation_3x_49_epochs.ckpt --reference_path /home/kin/fastflow3d.ckpt --output_path /home/kin/zeroflow3x.ckpt
```

- model_path, you can download from: [kylevedder/zeroflow_weights](https://github.com/kylevedder/zeroflow_weights/tree/master/argo)
- reference_path,  you can download fastflow3d model from: [zendo](https://zenodo.org/records/12632962)
- output_path, the converted model path. You can then run any evaluation script and visualization script with the converted model.