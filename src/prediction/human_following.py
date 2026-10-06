import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "Data"
)

TERRAIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "model",
    "random_forest_model.pkl"
)

HUMAN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "model",
    "human_random_forest_model.pkl"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading trained models...")

terrain_model = joblib.load(
    TERRAIN_MODEL_PATH
)

human_model = joblib.load(
    HUMAN_MODEL_PATH
)

print("Models loaded successfully.")


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    # Same resolution used during training
    image = image.resize(
        (496, 279)
    )

    image_array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    # Create 3x3 RGB neighbourhoods
    patches = sliding_window_view(
        image_array,
        (3, 3),
        axis=(0, 1)
    )

    # 3 x 3 x 3 = 27 features
    X = patches.reshape(
        -1,
        27
    )

    return image, X


# ============================================================
# HUMAN DETECTION
# ============================================================

def detect_human(image_path):

    image, X = prepare_image(
        image_path
    )

    # Predict human/non-human pixels
    prediction = human_model.predict(
        X
    )

    prediction_mask = prediction.reshape(
        image.size[1] - 2,
        image.size[0] - 2
    )

    # Convert mask to OpenCV format
    mask_uint8 = (
        prediction_mask.astype(np.uint8)
        * 255
    )

    # Find connected regions
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
    )

    # Ignore very small regions
    MIN_AREA = 50

    valid_components = []

    for component_id in range(
        1,
        num_labels
    ):

        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_AREA:

            valid_components.append(
                component_id
            )

    # No human detected
    if not valid_components:

        return {
            "detected": False,
            "position": "UNKNOWN",
            "bbox": None,
            "center": None,
            "mask": prediction_mask
        }

    # Select largest detected component
    human_component_id = max(
        valid_components,
        key=lambda i:
        stats[
            i,
            cv2.CC_STAT_AREA
        ]
    )

    # Bounding box
    x = stats[
        human_component_id,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        human_component_id,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        human_component_id,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        human_component_id,
        cv2.CC_STAT_HEIGHT
    ]

    # --------------------------------------------------------
    # Convert mask coordinates to 496 x 279 coordinates
    # --------------------------------------------------------

    mask_width = image.width - 2
    mask_height = image.height - 2

    scale_x = image.width / mask_width
    scale_y = image.height / mask_height

    x_original = x * scale_x
    y_original = y * scale_y

    w_original = w * scale_x
    h_original = h * scale_y

    # Human center
    center_x = (
        x_original
        + w_original / 2
    )

    center_y = (
        y_original
        + h_original / 2
    )

    # --------------------------------------------------------
    # Human LEFT / CENTER / RIGHT
    # --------------------------------------------------------

    if center_x < image.width / 3:

        position = "LEFT"

    elif center_x < (
        2 * image.width / 3
    ):

        position = "CENTER"

    else:

        position = "RIGHT"

    return {
        "detected": True,
        "position": position,
        "bbox": (
            x_original,
            y_original,
            w_original,
            h_original
        ),
        "center": (
            center_x,
            center_y
        ),
        "mask": prediction_mask
    }


# ============================================================
# TERRAIN ANALYSIS
# ============================================================

def analyze_terrain(image_path):

    image, X = prepare_image(
        image_path
    )

    # Predict traversable/non-traversable pixels
    prediction = terrain_model.predict(
        X
    )

    terrain_mask = prediction.reshape(
        image.size[1] - 2,
        image.size[0] - 2
    )

    height, width = terrain_mask.shape

    # Use lower 45% of image for navigation
    start_row = int(
        height * 0.55
    )

    navigation_area = terrain_mask[
        start_row:,
        :
    ]

    # Divide navigation area into 3 regions
    left_region = navigation_area[
        :,
        :width // 3
    ]

    center_region = navigation_area[
        :,
        width // 3:2 * width // 3
    ]

    right_region = navigation_area[
        :,
        2 * width // 3:
    ]

    # Calculate safety percentages
    left_score = (
        np.mean(left_region)
        * 100
    )

    center_score = (
        np.mean(center_region)
        * 100
    )

    right_score = (
        np.mean(right_region)
        * 100
    )

    return {
        "mask": terrain_mask,
        "left": left_score,
        "center": center_score,
        "right": right_score
    }


# ============================================================
# FOLLOWING DECISION
# ============================================================

def make_decision(
    human_position,
    left_score,
    center_score,
    right_score,
    safe_threshold=60
):

    if human_position == "LEFT":

        if left_score >= safe_threshold:
            return "FOLLOW LEFT"

        return "STOP"

    elif human_position == "CENTER":

        if center_score >= safe_threshold:
            return "FOLLOW FORWARD"

        return "STOP"

    elif human_position == "RIGHT":

        if right_score >= safe_threshold:
            return "FOLLOW RIGHT"

        return "STOP"

    return "SEARCH"


# ============================================================
# VISUAL RESULT
# ============================================================

def show_result(
    image,
    human_result,
    terrain_result,
    decision,
    image_name
):

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    # Display image
    ax.imshow(image)

    # --------------------------------------------------------
    # Human bounding box
    # --------------------------------------------------------

    if human_result["detected"]:

        x, y, w, h = (
            human_result["bbox"]
        )

        rectangle = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=3,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(
            rectangle
        )

        # Human center
        center_x, center_y = (
            human_result["center"]
        )

        ax.scatter(
            center_x,
            center_y,
            s=100,
            marker="x",
            linewidths=3
        )

        # Human label
        ax.text(
            x,
            max(5, y - 5),
            f"HUMAN - "
            f"{human_result['position']}",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8
            )
        )

    # --------------------------------------------------------
    # Navigation region boundaries
    # --------------------------------------------------------

    ax.axvline(
        496 / 3,
        linestyle="--",
        linewidth=1.5
    )

    ax.axvline(
        2 * 496 / 3,
        linestyle="--",
        linewidth=1.5
    )

    # Start of navigation area
    ax.axhline(
        279 * 0.55,
        linestyle="--",
        linewidth=1.5
    )

    # --------------------------------------------------------
    # Left safety score
    # --------------------------------------------------------

    ax.text(
        496 / 6,
        260,
        f"LEFT\n"
        f"{terrain_result['left']:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.75
        )
    )

    # --------------------------------------------------------
    # Center safety score
    # --------------------------------------------------------

    ax.text(
        496 / 2,
        260,
        f"CENTER\n"
        f"{terrain_result['center']:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.75
        )
    )

    # --------------------------------------------------------
    # Right safety score
    # --------------------------------------------------------

    ax.text(
        5 * 496 / 6,
        260,
        f"RIGHT\n"
        f"{terrain_result['right']:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.75
        )
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    ax.text(
        10,
        20,
        f"FINAL DECISION: {decision}",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9
        )
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"Autonomous Human-Following Robot\n"
        f"{image_name}",
        fontsize=16,
        fontweight="bold"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN HUMAN-FOLLOWING PIPELINE
# ============================================================

def run_human_following(
    image_name,
    show_image=True
):

    print("=" * 60)
    print("AUTONOMOUS HUMAN-FOLLOWING SYSTEM")
    print("=" * 60)

    print(f"\nImage: {image_name}")

    # --------------------------------------------------------
    # Create image path
    # --------------------------------------------------------

    image_path = os.path.join(
        DATA_DIR,
        image_name
    )

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image for visualization
    # --------------------------------------------------------

    display_image = Image.open(
        image_path
    ).convert("RGB")

    display_image = display_image.resize(
        (496, 279)
    )

    # --------------------------------------------------------
    # HUMAN DETECTION
    # --------------------------------------------------------

    human_result = detect_human(
        image_path
    )

    # --------------------------------------------------------
    # TERRAIN ANALYSIS
    # --------------------------------------------------------

    terrain_result = analyze_terrain(
        image_path
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    decision = make_decision(
        human_result["position"],
        terrain_result["left"],
        terrain_result["center"],
        terrain_result["right"]
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nHuman Detection")
    print("----------------")

    if human_result["detected"]:

        x, y, w, h = (
            human_result["bbox"]
        )

        center_x, center_y = (
            human_result["center"]
        )

        print(
            f"Bounding box: "
            f"x={x:.1f}, "
            f"y={y:.1f}, "
            f"w={w:.1f}, "
            f"h={h:.1f}"
        )

        print(
            f"Human center: "
            f"({center_x:.1f}, "
            f"{center_y:.1f})"
        )

        print(
            f"Human position: "
            f"{human_result['position']}"
        )

    else:

        print("No human detected.")

    print("\nTerrain Safety")
    print("----------------")

    print(
        f"Left:   "
        f"{terrain_result['left']:.2f}%"
    )

    print(
        f"Center: "
        f"{terrain_result['center']:.2f}%"
    )

    print(
        f"Right:  "
        f"{terrain_result['right']:.2f}%"
    )

    print("\nFinal Decision")
    print("----------------")

    print(decision)

    print("=" * 60)

    # ========================================================
    # SHOW IMAGE
    # ========================================================

    if show_image:

        show_result(
            display_image,
            human_result,
            terrain_result,
            decision,
            image_name
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "image": image_name,

        "human_detected":
            human_result["detected"],

        "human_position":
            human_result["position"],

        "human_bbox":
            human_result["bbox"],

        "human_center":
            human_result["center"],

        "left_score":
            terrain_result["left"],

        "center_score":
            terrain_result["center"],

        "right_score":
            terrain_result["right"],

        "decision":
            decision
    }


# ============================================================
# RUN FROM TERMINAL
# ============================================================

if __name__ == "__main__":

    run_human_following(
        "left25057.jpg"
    )