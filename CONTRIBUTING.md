# Contributing to [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow)

We want to make contributing to this project as easy and transparent as possible. We welcome any contributions, from bug fixes to new features. If you're interested in adding your own scene flow method, this guide will walk you through the process.

## Adding a New Method

Here is a quick guide to integrating a new method into the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) codebase.

### 1. Data Preparation

All data is expected to be processed into the `.h5` format. Each file represents a scene, and within the file, each data sample is indexed by a `timestamp` key.

For more details on the data processing pipeline, please see the [Data Processing README](./dataprocess/README.md#process).

### 2. Model Implementation

All model source files are located in [`src/models`](./src/models). When adding your model, please remember to import your new model class in the [`src/models/__init__.py`](./src/models/__init__.py) file. Don't forget to add your model conf files in [`conf/model`](./conf/model).

  * **For Feed-Forward Methods:** You can use `deflow` and `fastflow3d` as implementation examples.
  * **For Optimization-Based Methods:** Please refer to `nsfp` and `fastnsf` for guidance on structure and integration. A detailed example can be found in the [NSFP model file](./src/models/nsfp.py).

### 3. Custom Loss Functions

All loss functions are defined in [`src/lossfuncs.py`](./src/lossfuncs.py). If your model requires a new loss function, you can add it to this file by following the pattern of the existing functions. SeFlow provided a self-supervised loss function example for all feed-forward methods. Feel free to check.

### 4.1 Training a Feed-Forward Model

1.  Add a configuration file for your new model in the [`conf/model`](./conf/model) directory.
2.  Begin training by running the following command:
    ```bash
    python train.py model=your_model_name
    ```
3.  **Note:** If your model's output dictionary (`res_dict`) has a different structure from the existing models, you may need to add a new pattern in the `training_step` and `validation_step` methods in the main training script.

### 4.2 Running an Optimization-Based Model

Our framework supports multi-GPU execution for optimization-based methods out of the box. You can follow the structure of existing methods like NSFP to run and evaluate your model.

-----

Once the steps above are completed, other parts of the framework, such as evaluation (`eval`) and visualization (`save`), should integrate with your new model accordingly.

Thank you for your contribution!