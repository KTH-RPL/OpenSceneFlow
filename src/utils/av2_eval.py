import zipfile
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from rich.progress import track

"""Constants for scene flow evaluation."""

from enum import Enum, unique
from typing import Final

from av2.datasets.sensor.constants import AnnotationCategories

SCENE_FLOW_DYNAMIC_THRESHOLD: Final = 0.05
SWEEP_PAIR_TIME_DELTA: Final = 0.1
CLOSE_DISTANCE_THRESHOLD: Final = 35.0

CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}
# These catagories are ignored because of labeling oddities (moving stop signs on side of school bus, etc)
ROAD_SIGNS = [
    "BOLLARD",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "SIGN",
    "STOP_SIGN",
    "MESSAGE_BOARD_TRAILER",
    "TRAFFIC_LIGHT_TRAILER",
]
PEDESTRIAN_CATEGORIES = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]
WHEELED_VRU = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]
CAR = ["REGULAR_VEHICLE"]
OTHER_VEHICLES = [
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "ARTICULATED_BUS",
    "BUS",
    "SCHOOL_BUS",
]
BACKGROUND_CATEGORIES = ["NONE"] # + ROAD_SIGNS
BUCKETED_METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "CAR": CAR,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "WHEELED_VRU": WHEELED_VRU,
    "OTHER_VEHICLES": OTHER_VEHICLES,
}

import av2.geometry.geometry as geometry_utils
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat
from av2.utils.io import read_feather

# Mapping from egovehicle time in nanoseconds to egovehicle pose.
TimestampedCitySE3EgoPoses = Dict[int, SE3]

# Mapping from sensor name to sensor pose.
SensorPosesMapping = Dict[str, SE3]

def read_ego_SE3_sensor(log_dir: Path) -> SensorPosesMapping:
    """Read the sensor poses for the given log.

    The sensor pose defines an SE3 transformation from the sensor reference frame to the egovehicle reference frame.
    Mathematically we define this transformation as: $$ego_SE3_sensor$$.

    In other words, when this transformation is applied to a set of points in the sensor reference frame, they
    will be transformed to the egovehicle reference frame.

    Example (1).
        points_ego = ego_SE3_sensor(points_sensor) apply the SE3 transformation to points in the sensor reference frame.

    Example (2).
        sensor_SE3_ego = ego_SE3_sensor^{-1} take the inverse of the SE3 transformation.
        points_sensor = sensor_SE3_ego(points_ego) apply the SE3 transformation to points in the ego reference frame.

    Extrinsics:
        sensor_name: Name of the sensor.
        qw: scalar component of a quaternion.
        qx: X-axis coefficient of a quaternion.
        qy: Y-axis coefficient of a quaternion.
        qz: Z-axis coefficient of a quaternion.
        tx_m: X-axis translation component.
        ty_m: Y-axis translation component.
        tz_m: Z-axis translation component.

    Args:
        log_dir: Path to the log directory.

    Returns:
        Mapping from sensor name to sensor pose.
    """
    ego_SE3_sensor_path = Path(log_dir, "calibration", "egovehicle_SE3_sensor.feather")
    ego_SE3_sensor = read_feather(ego_SE3_sensor_path)
    rotations = geometry_utils.quat_to_mat(
        ego_SE3_sensor.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy()
    )
    translations = ego_SE3_sensor.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
    sensor_names = ego_SE3_sensor.loc[:, "sensor_name"].to_numpy()

    sensor_name_to_pose: SensorPosesMapping = {
        name: SE3(rotation=rotations[i], translation=translations[i])
        for i, name in enumerate(sensor_names)
    }
    return sensor_name_to_pose

@unique
class SceneFlowMetricType(str, Enum):
    """Scene Flow metrics."""

    ACCURACY_RELAX = "ACCURACY_RELAX"
    ACCURACY_STRICT = "ACCURACY_STRICT"
    ANGLE_ERROR = "ANGLE_ERROR"
    EPE = "EPE"


@unique
class SegmentationMetricType(str, Enum):
    """Segmentation metrics."""

    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"


@unique
class InanimateCategories(str, Enum):
    """Annotation categories representing inanimate objects that aren't vehicles."""

    BOLLARD = "BOLLARD"
    CONSTRUCTION_BARREL = "CONSTRUCTION_BARREL"
    CONSTRUCTION_CONE = "CONSTRUCTION_CONE"
    MOBILE_PEDESTRIAN_CROSSING_SIGN = "MOBILE_PEDESTRIAN_CROSSING_SIGN"
    SIGN = "SIGN"
    STOP_SIGN = "STOP_SIGN"


@unique
class LeggedCategories(str, Enum):
    """Annotation categories representing objects that move using legs."""

    ANIMAL = "ANIMAL"
    DOG = "DOG"
    OFFICIAL_SIGNALER = "OFFICIAL_SIGNALER"
    PEDESTRIAN = "PEDESTRIAN"


@unique
class SmallVehicleCategories(str, Enum):
    """Annotation categories representing small vehicles."""

    BICYCLE = "BICYCLE"
    BICYCLIST = "BICYCLIST"
    MOTORCYCLE = "MOTORCYCLE"
    MOTORCYCLIST = "MOTORCYCLIST"
    STROLLER = "STROLLER"
    WHEELCHAIR = "WHEELCHAIR"
    WHEELED_DEVICE = "WHEELED_DEVICE"
    WHEELED_RIDER = "WHEELED_RIDER"


@unique
class VehicleCategories(str, Enum):
    """Annotation categories representing regular vehicles."""

    ARTICULATED_BUS = "ARTICULATED_BUS"
    BOX_TRUCK = "BOX_TRUCK"
    BUS = "BUS"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    MESSAGE_BOARD_TRAILER = "MESSAGE_BOARD_TRAILER"
    RAILED_VEHICLE = "RAILED_VEHICLE"
    REGULAR_VEHICLE = "REGULAR_VEHICLE"
    SCHOOL_BUS = "SCHOOL_BUS"
    TRAFFIC_LIGHT_TRAILER = "TRAFFIC_LIGHT_TRAILER"
    TRUCK = "TRUCK"
    TRUCK_CAB = "TRUCK_CAB"
    VEHICULAR_TRAILER = "VEHICULAR_TRAILER"


@unique
class MetricBreakdownCategories(str, Enum):
    """Meta-categories for the scene flow task."""

    ALL = "All"
    BACKGROUND = "Background"
    FOREGROUND = "Foreground"


NO_CLASS_BREAKDOWN: Final = {MetricBreakdownCategories.ALL: list(range(31))}
FOREGROUND_BACKGROUND_BREAKDOWN: Final = {
    MetricBreakdownCategories.BACKGROUND: [0],
    MetricBreakdownCategories.FOREGROUND: [
        CATEGORY_TO_INDEX[k.value]
        for k in (
            list(InanimateCategories)
            + list(LeggedCategories)
            + list(SmallVehicleCategories)
            + list(VehicleCategories)
        )
    ],
}

FLOW_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m")


from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

ACCURACY_RELAX_DISTANCE_THRESHOLD: Final = 0.1
ACCURACY_STRICT_DISTANCE_THRESHOLD: Final = 0.05
NO_FMT_INDICES: Final = ("Background", "Dynamic")
EPS: Final = 1e-10

def compute_end_point_error(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the end-point error between predictions and ground truth.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The point-wise end-point error.
    """
    end_point_error: NDArrayFloat = np.linalg.norm(dts - gts, axis=-1).astype(
        np.float64
    )
    return end_point_error


def compute_accuracy(
    dts: NDArrayFloat, gts: NDArrayFloat, distance_threshold: float
) -> NDArrayFloat:
    """Compute the percent of inliers for a given threshold for a set of prediction and ground truth vectors.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.
        distance_threshold: Distance threshold for classifying inliers.

    Returns:
        The pointwise inlier assignments.
    """
    l2_norm = np.linalg.norm(dts - gts, axis=-1)
    gts_norm = np.linalg.norm(gts, axis=-1)
    relative_error = np.divide(l2_norm, gts_norm + EPS)
    abs_error_inlier = np.less(l2_norm, distance_threshold).astype(bool)
    relative_error_inlier = np.less(relative_error, distance_threshold).astype(bool)
    accuracy: NDArrayFloat = np.logical_or(
        abs_error_inlier, relative_error_inlier
    ).astype(np.float64)
    return accuracy


def compute_accuracy_strict(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the accuracy with a 0.05 threshold.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise inlier assignments at a 0.05 threshold
    """
    return compute_accuracy(dts, gts, ACCURACY_STRICT_DISTANCE_THRESHOLD)


def compute_accuracy_relax(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the accuracy with a 0.1 threshold.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise inlier assignments at a 0.1 threshold.
    """
    return compute_accuracy(dts, gts, ACCURACY_RELAX_DISTANCE_THRESHOLD)


def compute_angle_error(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the angle error in space-time between the prediced and ground truth flow vectors.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise angle errors in space-time.
    """
    # Convert the 3D flow vectors to 4D space-time vectors.
    dts_space_time = np.pad(
        dts, ((0, 0), (0, 1)), constant_values=SWEEP_PAIR_TIME_DELTA
    )
    gts_space_time = np.pad(
        gts, ((0, 0), (0, 1)), constant_values=SWEEP_PAIR_TIME_DELTA
    )

    dts_space_time_norm = np.linalg.norm(dts_space_time, axis=-1, keepdims=True)
    gts_space_time_norm = np.linalg.norm(gts_space_time, axis=-1, keepdims=True)
    unit_dts = dts_space_time / dts_space_time_norm
    unit_gts = gts_space_time / gts_space_time_norm

    dot_product = np.einsum("bd,bd->b", unit_dts, unit_gts)

    # Floating point errors can cause `dot_product` to be slightly greater than 1 or less than -1.
    clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_error: NDArrayFloat = np.arccos(clipped_dot_product).astype(np.float64)
    return angle_error


def compute_true_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true positive count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of true positive classifications.
    """
    return int(np.logical_and(dts, gts).sum())


def compute_true_negatives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true negative count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of true negative classifications.
    """
    return int(np.logical_and(~dts, ~gts).sum())


def compute_false_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute false positive count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of false positive classifications.
    """
    return int(np.logical_and(dts, ~gts).sum())


def compute_false_negatives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute false negative count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of false negative classifications
    """
    return int(np.logical_and(~dts, gts).sum())


def compute_scene_flow_metrics(
    dts: NDArrayFloat, gts: NDArrayFloat, scene_flow_metric_type: SceneFlowMetricType
) -> NDArrayFloat:
    """Compute scene flow metrics.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.
        scene_flow_metric_type: Scene flow metric type.

    Returns:
        Scene flow metric corresponding to `scene_flow_metric_type`.

    Raises:
        NotImplementedError: If the `scene_flow_metric_type` is not implemented.
    """
    if scene_flow_metric_type == SceneFlowMetricType.ACCURACY_RELAX:
        return compute_accuracy_relax(dts, gts)
    elif scene_flow_metric_type == SceneFlowMetricType.ACCURACY_STRICT:
        return compute_accuracy_strict(dts, gts)
    elif scene_flow_metric_type == SceneFlowMetricType.ANGLE_ERROR:
        return compute_angle_error(dts, gts)
    elif scene_flow_metric_type == SceneFlowMetricType.EPE:
        return compute_end_point_error(dts, gts)
    else:
        raise NotImplementedError(
            f"The scene flow metric type {scene_flow_metric_type} is not implemented!"
        )


def compute_segmentation_metrics(
    dts: NDArrayBool, gts: NDArrayBool, segmentation_metric_type: SegmentationMetricType
) -> int:
    """Compute segmentation metrics.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.
        segmentation_metric_type: Segmentation metric type.

    Returns:
        Segmentation metric corresponding to `segmentation_metric_type`.

    Raises:
        NotImplementedError: If the `segmentation_metric_type` is not implemented.
    """
    if segmentation_metric_type == SegmentationMetricType.TP:
        return compute_true_positives(dts, gts)
    elif segmentation_metric_type == SegmentationMetricType.TN:
        return compute_true_negatives(dts, gts)
    elif segmentation_metric_type == SegmentationMetricType.FP:
        return compute_false_positives(dts, gts)
    elif segmentation_metric_type == SegmentationMetricType.FN:
        return compute_false_negatives(dts, gts)
    else:
        raise NotImplementedError(
            f"The segmentation metric type {segmentation_metric_type} is not implemented!"
        )

EPS: Final = 1e-6
def compute_epe(res_dict, indices, eps=1e-8):
    epe_sum = 0
    count_sum = 0
    for index in indices:
        count = res_dict['Count'][index]
        if count != 0:
            epe_sum += res_dict['EPE'][index] * count
            count_sum += count
    return epe_sum / (count_sum + eps) if count_sum != 0 else 0.0

def compute_metrics(
    pred_flow: NDArrayFloat,
    pred_dynamic: NDArrayBool,
    gts: NDArrayFloat,
    category_indices: NDArrayInt,
    is_dynamic: NDArrayBool,
    is_close: NDArrayBool,
    is_valid: NDArrayBool,
    # metric_categories: Dict[MetricBreakdownCategories, List[int]],
) -> Dict[str, List[Any]]:
    """Compute all the metrics for a given example and package them into a list to be put into a DataFrame.

    Args:
        pred_flow: (N,3) Predicted flow vectors.
        pred_dynamic: (N,) Predicted dynamic labels.
        gts: (N,3) Ground truth flow vectors.
        category_indices: (N,) Integer class labels for each point.
        is_dynamic: (N,) Ground truth dynamic labels.
        is_close: (N,) True for a point if it is within a 70m x 70m box around the AV.
        is_valid: (N,) True for a point if its flow vector was successfully computed.
        metric_categories: A dictionary mapping segmentation labels to groups of category indices.

    Returns:
        A dictionary of columns to create a long-form DataFrame of the results from.
        One row for each subset in the breakdown.
    """
    metric_categories = FOREGROUND_BACKGROUND_BREAKDOWN
    pred_flow = pred_flow[is_valid].astype(np.float64)
    pred_dynamic = pred_dynamic[is_valid].astype(bool)
    gts = gts[is_valid].astype(np.float64)
    category_indices = category_indices[is_valid].astype(int)
    is_dynamic = is_dynamic[is_valid].astype(bool)
    is_close = is_close[is_valid].astype(bool)

    results: DefaultDict[str, List[Any]] = defaultdict(list)

    # Each metric is broken down by point labels on Object Class, Motion, and Distance from the AV.
    # We iterate over all combinations of those three categories and compute average metrics on each subset.
    for cls, category_idxs in metric_categories.items():
        # Compute the union of all masks within the meta-category.
        category_mask = category_indices == category_idxs[0]
        for i in category_idxs[1:]:
            category_mask = np.logical_or(category_mask, (category_indices == i))

        for motion, m_mask in [("Dynamic", is_dynamic), ("Static", ~is_dynamic)]:
            for distance, d_mask in [("Close", is_close), ("Far", ~is_close)]:
                mask = category_mask & m_mask & d_mask
                subset_size = mask.sum().item()
                gts_sub = gts[mask]
                pred_sub = pred_flow[mask]
                results["Class"] += [cls.value]
                results["Motion"] += [motion]
                results["Distance"] += [distance]
                results["Count"] += [subset_size]

                # Check if there are any points in this subset and if so compute all the average metrics.
                if subset_size > 0:
                    for flow_metric_type in SceneFlowMetricType:
                        results[flow_metric_type] += [
                            compute_scene_flow_metrics(
                                pred_sub, gts_sub, flow_metric_type
                            ).mean()
                        ]
                    for seg_metric_type in SegmentationMetricType:
                        results[seg_metric_type] += [
                            compute_segmentation_metrics(
                                pred_dynamic[mask], is_dynamic[mask], seg_metric_type
                            )
                        ]
                else:
                    for flow_metric_type in SceneFlowMetricType:
                        results[flow_metric_type] += [np.nan]
                    for seg_metric_type in SegmentationMetricType:
                        results[seg_metric_type] += [0.0]

    # reference: eval.py L503: https://github.com/argoverse/av2-api/blob/main/src/av2/evaluation/scene_flow/eval.py
    # we need Dynamic IoU and EPE 3-Way Average to calculate loss!
    # weighted: (x[metric_type.value] * x.Count).sum() / total
    # 'Class': ['Background', 'Background', 'Background', 'Background', 'Foreground', 'Foreground', 'Foreground', 'Foreground']
    # 'Motion': ['Dynamic', 'Dynamic', 'Static', 'Static', 'Dynamic', 'Dynamic', 'Static', 'Static']
    # 'Distance': ['Close', 'Far', 'Close', 'Far', 'Close', 'Far', 'Close', 'Far']
    
    EPE_Background_Static = compute_epe(results, [2, 3])
    EPE_Dynamic = compute_epe(results, [4, 5])
    EPE_Foreground_Static = compute_epe(results, [6, 7])
    Dynamic_IoU = sum(results['TP']) / (sum(results['TP']) + sum(results['FP']) + sum(results['FN'])+EPS)

    return {
        'EPE_BS': EPE_Background_Static, 
        'EPE_FD': EPE_Dynamic, 
        'EPE_FS': EPE_Foreground_Static, 
        'IoU': Dynamic_IoU
    }


def evaluate_predictions(
    annotations_dir: Path, get_prediction: Callable[[Path], pd.DataFrame]
) -> pd.DataFrame:
    """Run the evaluation on predictions and labels.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        get_prediction: Function that retrieves a predictions DataFrame for a given relative
                        annotation filepath, or None if no prediction exists.

    Returns:
        DataFrame containing the average metrics on each subset of each example.
    """
    results: DefaultDict[str, List[Any]] = defaultdict(list)
    annotation_files = sorted(annotations_dir.rglob("*.feather"))
    for anno_file in track(annotation_files, description="Evaluating..."):
        gts = pd.read_feather(anno_file)
        name: Path = anno_file.relative_to(annotations_dir)
        pred = get_prediction(name)
        if pred is None:
            continue
        current_example_results = compute_metrics(
            pred[list(FLOW_COLUMNS)].to_numpy().astype(float),
            pred["is_dynamic"].to_numpy().astype(bool),
            gts[list(FLOW_COLUMNS)].to_numpy().astype(float),
            gts["category_indices"].to_numpy().astype(np.uint8),
            gts["is_dynamic"].to_numpy().astype(bool),
            gts["is_close"].to_numpy().astype(bool),
            gts["is_valid"].to_numpy().astype(bool),
            FOREGROUND_BACKGROUND_BREAKDOWN,
        )
        num_subsets = len(list(current_example_results.values())[0])
        results["Example"] += [str(name) for _ in range(num_subsets)]
        for m in current_example_results:
            results[m] += current_example_results[m]
    df = pd.DataFrame(
        results,
        columns=["Example", "Class", "Motion", "Distance", "Count"]
        + list(SceneFlowMetricType)
        + list(SegmentationMetricType),
    )
    return df


def get_prediction_from_directory(
    annotation_name: Path, predictions_dir: Path
) -> Optional[pd.DataFrame]:
    """Get the prediction corresponding annotation from a directory of prediction files.

    Args:
        annotation_name: Relative path to the annotation file.
        predictions_dir: Path to the predicition files in submission_format.

    Returns:
        DataFrame containing the predictions for that annotation file or None if it does not exist.
    """
    pred_file = predictions_dir / annotation_name
    if not pred_file.exists():
        return None
    pred = pd.read_feather(pred_file)
    return pred


def get_prediction_from_zipfile(
    annotation_name: Path, predictions_zip: Path
) -> Optional[pd.DataFrame]:
    """Get the prediction corresponding annotation from a zip archive of prediction files.

    Args:
        annotation_name: Relative path to the annotation file.
        predictions_zip: Path to the prediction files in a zip archive.

    Returns:
        DataFrame containing the predictions for that annotation file or None if it does not exist.
    """
    with ZipFile(predictions_zip, "r") as zf:
        name = annotation_name.as_posix()
        path = zipfile.Path(zf, name)
        if path.exists():
            return pd.read_feather(zf.open(name))
        else:
            return None

def evaluate_directories(annotations_dir: Path, predictions_dir: Path) -> pd.DataFrame:
    """Run the evaluation on predictions and labels saved to disk.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.

    Returns:
        DataFrame containing the average metrics on each subset of each example.
    """
    return evaluate_predictions(
        annotations_dir, lambda n: get_prediction_from_directory(n, predictions_dir)
    )

def evaluate_zip(annotations_dir: Path, predictions_zip: Path) -> pd.DataFrame:
    """Run the evaluation on predictions and labels saved to disk.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_zip: Path to the prediction files in a zip archive.

    Returns:
        DataFrame containing the average metrics on each subset of each example.
    """
    return evaluate_predictions(
        annotations_dir, lambda n: get_prediction_from_zipfile(n, predictions_zip)
    )

def results_to_dict(frame: pd.DataFrame) -> Dict[str, float]:
    """Convert a results DataFrame to a dictionary of whole dataset metrics.

    Args:
        frame: DataFrame returned by evaluate_directories.

    Returns:
        Dictionary string keys "<Motion/Class/Distance/Metric>" mapped to average metrics on that subset.
    """
    output = {}
    grouped = frame.groupby(["Class", "Motion", "Distance"])

    def weighted_average(
        x: pd.DataFrame, metric_type: Union[SceneFlowMetricType, SegmentationMetricType]
    ) -> float:
        """Weighted average of metric m using the Count column.

        Args:
            x: Input data-frame.
            metric_type: Metric type.

        Returns:
            Weighted average over the metric_type;
        """
        total = cast(int, x["Count"].sum())
        if total == 0:
            return np.nan
        averages: float = (x[metric_type.value] * x.Count).sum() / total
        return averages

    for metric_type in SceneFlowMetricType:
        avg: pd.Series[float] = grouped.apply(
            lambda x, m=metric_type: weighted_average(x, metric_type=m)
        )
        segments: List[Tuple[str, str, str]] = avg.index.to_list()
        for segment in segments:
            if segment[:2] == NO_FMT_INDICES:
                continue

            metric_type_str = (
                metric_type.title().replace("_", " ")
                if metric_type != SceneFlowMetricType.EPE
                else metric_type
            )
            name = metric_type_str + "/" + "/".join([str(i) for i in segment])
            output[name] = avg.loc[segment]

    grouped = frame.groupby(["Class", "Motion"])
    for metric_type in SceneFlowMetricType:
        avg_nodist: pd.Series[float] = grouped.apply(
            lambda x, m=metric_type: weighted_average(x, metric_type=m)
        )
        segments_nodist: List[Tuple[str, str, str]] = avg_nodist.index.to_list()
        for segment in segments_nodist:
            if segment[:2] == NO_FMT_INDICES:
                continue

            metric_type_str = (
                metric_type.title().replace("_", " ")
                if metric_type != SceneFlowMetricType.EPE
                else metric_type
            )
            name = metric_type_str + "/" + "/".join([str(i) for i in segment])
            output[name] = avg_nodist.loc[segment]
    output["Dynamic IoU"] = frame.TP.sum() / (
        frame.TP.sum() + frame.FP.sum() + frame.FN.sum()
    )
    output["EPE 3-Way Average"] = (
        output["EPE/Foreground/Dynamic"]
        + output["EPE/Foreground/Static"]
        + output["EPE/Background/Static"]
    ) / 3
    return output

def evaluate(annotations_dir: str, predictions_dir: str) -> Dict[str, float]:
    """Evaluate a set of predictions and print the results.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.

    Returns:
        The results as a dict of metric names and values.
    """
    results_df = evaluate_directories(Path(annotations_dir), Path(predictions_dir))
    results_dict = results_to_dict(results_df)

    for metric in sorted(results_dict):
        print(f"{metric}: {results_dict[metric]:.3f}")

    return results_dict

def write_output_file(
    flow: NDArrayFloat,
    is_dynamic: NDArrayBool,
    sweep_uuid: Tuple[str, int],
    output_dir: Path,
    leaderboard_version: int = 1,
) -> None:
    """Write an output predictions file in the correct format for submission.

    Args:
        flow: (N,3) Flow predictions.
        is_dynamic: (N,) Dynamic segmentation prediction.
        sweep_uuid: Identifier of the sweep being predicted (log_id, timestamp_ns).
        output_dir: Top level directory containing all predictions.
        leaderboard_version: Version of the leaderboard format to use.
            version 1 for: https://eval.ai/web/challenges/challenge-page/2010/evaluation
            version 2 for: https://eval.ai/web/challenges/challenge-page/2210/evaluation
    """
    output_log_dir = output_dir / sweep_uuid[0]
    output_log_dir.mkdir(exist_ok=True, parents=True)
    fx_m = flow[:, 0].astype(np.float16)
    fy_m = flow[:, 1].astype(np.float16)
    fz_m = flow[:, 2].astype(np.float16)
    if leaderboard_version == 1:

        output = pd.DataFrame(
            {
                "flow_tx_m": fx_m,
                "flow_ty_m": fy_m,
                "flow_tz_m": fz_m,
                "is_dynamic": is_dynamic.astype(bool),
            }
        )
        output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")
    elif leaderboard_version == 2:
        output = pd.DataFrame(
            {
                "is_valid": np.ones_like(fx_m, dtype=bool),
                "flow_tx_m": fx_m,
                "flow_ty_m": fy_m,
                "flow_tz_m": fz_m,
            }
        )
        output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")
    
from zipfile import ZipFile
from torch import BoolTensor
import torch
def get_eval_point_mask(sweep_uuid: Tuple[str, int], mask_file: Path) -> BoolTensor:
    """Retrieve for a given sweep, a boolean mask indicating which points are evaluated on.

    Args:
        sweep_uuid: The uuid of the first sweep in the pair to retrieve the mask for.
        mask_file: Archive of submission masks.

    Returns:
        The submission mask for that pair.
    """
    with ZipFile(mask_file) as masks:
        log_id, timestamp_ns = sweep_uuid
        mask = (
            pd.read_feather(masks.open(f"{log_id}/{timestamp_ns}.feather"))
            .to_numpy()
            .astype(bool)
        )

    return BoolTensor(torch.from_numpy(mask).squeeze())


# python >= 3.7
from dataclasses import dataclass
@dataclass(frozen=True, eq=True, repr=True)
class BaseSplitValue:
    name: str
    avg_epe: float
    avg_range: float
    thresholds_range: Tuple[float, float]
    count: int
    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)
    
def compute_bucketed_epe(
    pred_flow: NDArrayFloat,
    gt_flow: NDArrayFloat,
    category_indices: NDArrayInt,
    is_valid: NDArrayBool,
):
    storage_error_matrix = []
    # bucket_max_speed, num_buckets, distance_thresholds set is from: eval/bucketed_epe.py#L226
    # HARCODE(Qingwen): if change it please search globally for the same value
    speed_splits = np.concatenate([np.linspace(0, 2.0, 51), [np.inf]])
    speed_thresholds = list(zip(speed_splits, speed_splits[1:]))

    gt_speeds = np.linalg.norm(gt_flow, axis=-1)
    error_flow = np.linalg.norm(pred_flow - gt_flow, axis=-1)
    # based on each category, compute the epe
    for cats_name in BUCKETED_METACATAGORIES:
        selected_classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats_name]]
        cat_mask = np.isin(category_indices, np.array(selected_classes_ids))
        # since background don't have speed, we just compute the average epe
        if cats_name == "BACKGROUND":
            mask = cat_mask & is_valid
            storage_error_matrix.append(BaseSplitValue(cats_name, error_flow[mask].mean(), gt_speeds[mask].mean(), (0.0, 0.04), mask.sum()))
            continue
        for min_speed_threshold, max_speed_threshold in speed_thresholds:
            speed_mask = (gt_speeds >= min_speed_threshold) & (gt_speeds < max_speed_threshold)
            mask = cat_mask & speed_mask & is_valid
            count_pts = mask.sum()
            if count_pts == 0:
                continue
            storage_error_matrix.append(BaseSplitValue(cats_name, error_flow[mask].mean(), gt_speeds[mask].mean(), (min_speed_threshold, max_speed_threshold), count_pts))
            
    return storage_error_matrix

def compute_ssf_metrics(
    pc0_dis: NDArrayFloat,
    pred_flow: NDArrayFloat,
    gt_flow: NDArrayFloat,
    is_valid: NDArrayBool,
    dynamic_speed = 1.4 # (m/s). since pedestrian walking speed normally is 1.4m/s
) -> Dict[str, List[Any]]:
    """
    Compute metrics based on distance for a given example and package them into a list to be put into a DataFrame.
    Args:
        pc0_dis: (N,3) point distance to ego vehicle.
        pred_flow: (N,3) Predicted flow vectors.
        gt_flow: (N,3) Ground truth flow vectors.
        is_valid: (N,) True for a point if its ground truth flow was successfully computed. Normally we will exclude the points belong to ground etc.
    Returns:
        A dictionary of columns to create a long-form DataFrame of the results from.
        One row for each subset in the breakdown.
    """
    storage_error_matrix = []
    # HARCODE(Qingwen): if change it please search globally for the same value
    distance_split = [0, 35, 50, 75, 100, np.inf]
    gt_speeds = np.linalg.norm(gt_flow, axis=-1) * 10 # since our sensor is 10Hz
    range_thresholds = list(zip(distance_split, distance_split[1:]))
    for min_range, max_range in range_thresholds:
        mask = (pc0_dis >= min_range) & (pc0_dis < max_range) & is_valid
        gt_speed_, pred_flow_, gt_flow_, pc0_dis_ = gt_speeds[mask], pred_flow[mask], gt_flow[mask], pc0_dis[mask]
        # static and dynamic split based on the ground truth speed
        dynamic_mask = gt_speed_ >= dynamic_speed 
        static_mask = ~dynamic_mask
        for motion, m_mask in [("Dynamic", dynamic_mask), ("Static", static_mask)]:
            count_pts = m_mask.sum()
            if count_pts == 0:
                continue
            epe = np.linalg.norm(pred_flow_ - gt_flow_, axis=-1)[m_mask].mean()
            avg_dis = pc0_dis_[m_mask].mean()
            storage_error_matrix.append(BaseSplitValue(motion, epe, avg_dis, (min_range, max_range), count_pts))
    return storage_error_matrix
