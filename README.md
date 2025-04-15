> 🚗 Official ROS package implementation of the paper:  
> **"NN-PL-VIO: A Customizable Neural Network Based VIO Framework with a Lightweight Point-Line Joint Network"**  

## Code Structure

- `camera_model`: Camera model definition files  
- `config`: Parameter configuration files  
- `feature_tracker`: Front-end feature processing module  
- `pose_graph`: Loop closure detection module  
- `vins_estimator`: Back-end pose estimation module  

## Usage

- Place all files into the `src` folder of your workspace, compile the workspace, and source the environment. Dependency installation can be referenced in `plvins.md`.
- Modify the parameter file and its content in the launch file according to your needs (by default, the system processes the EuRoC dataset).
- Launch the point-line feature processor:  

    ```bash
    roslaunch feature_tracker feature_tracker.launch  # Launch the point-line feature processor separately
    roslaunch feature_tracker plfeature_tracker.launch  # Launch the joint point-line feature processor, suitable for superplnet network
    ```

- Launch the back-end pose estimator and trajectory reconstruction:  

    ```bash
    roslaunch plvins_estimator estimator.launch  # The trajectory will be saved to the specified path after execution
    ```

For different datasets, please adjust the parameters in the `config` files accordingly, and specify the corresponding configuration in the launch files.

## Custom Front-End

The custom front-end includes point/line feature extraction and matching methods. Follow these steps to integrate your own methods:

1. Refer to `feature_tracker/scripts/utils_point/superpoint/model.py` and `feature_tracker/scripts/utils_line/sold2/model.py`.  
   Inherit from `BaseExtractModel` (contains the `extract` method) and `BaseMatchModel` (contains the `match` method) to implement your custom extraction and matching logic.
   
2. Write the instantiated class into  
   `feature_tracker/scripts/utils_point/my_point_model.py` and  
   `feature_tracker/scripts/utils_line/my_line_model.py`  
   following the provided format.

3. Add your method names and parameter definitions to the `config` files.

4. The system will automatically locate your custom method by name, instantiate it with the given parameters, and execute the customized front-end feature processor.

Predefined front-end methods include:

- **Point extraction**: `superpoint`, `orb`, `r2d2`  
- **Point matching**: `nnm`, `superglue`, `knn`  
- **Line extraction**: `sold2`, `lcnn`, `tplsd`  
- **Line matching**: `wunsch`  
- **Joint point-line inference**: `superpl-net`

⚠️ **Note:**  
When using the predefined point and line feature extraction or matching methods, please make sure to **download the corresponding pre-trained model weights** in advance, and specify the correct file paths in the config file.


## Acknowledgements

This project is inspired by and built upon the following excellent open-source repositories:

- [PL-VINS](https://github.com/PL-VINS/PL-VINS) — Point-Line Visual-Inertial Odometry.
- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) — Self-Supervised Interest Point Detector and Descriptor.
- [SOLD2](https://github.com/cvg/SOLD2) — Self-Supervised Line Detection and Description.

We sincerely thank the authors and contributors of these projects for making their work available to the community.
