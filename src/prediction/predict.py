import os
import joblib
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "model",
    "random_forest_model.pkl"
)


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)


# -----------------------------
# Prepare image
# -----------------------------
def prepare_image(image_path):

    image = Image.open(image_path).convert("RGB")

    # Model was trained using 496 x 279 images
    image = image.resize((496, 279))

    image_array = np.array(image).astype(np.float32) / 255.0

    # Create 3 x 3 pixel neighborhoods
    patches = sliding_window_view(
        image_array,
        (3, 3),
        axis=(0, 1)
    )

    X = patches.reshape(-1, 27)

    return image, X


# -----------------------------
# Predict traversability
# -----------------------------
def predict_traversability(image_path):

    image, X = prepare_image(image_path)

    prediction = model.predict(X)

    height = image.size[1]
    width = image.size[0]

    prediction_mask = prediction.reshape(
        height - 2,
        width - 2
    )

    return image, prediction_mask


# -----------------------------
# Navigation decision
# -----------------------------
def navigation_decision(prediction_mask):

    height, width = prediction_mask.shape

    # Use only the bottom 45% of the image
    # because this represents terrain closer
    # to the caddy
    start_row = int(height * 0.55)

    navigation_area = prediction_mask[start_row:, :]

    # Divide into three regions
    left = navigation_area[
        :,
        :width // 3
    ]

    center = navigation_area[
        :,
        width // 3:2 * width // 3
    ]

    right = navigation_area[
        :,
        2 * width // 3:
    ]

    # Calculate traversability percentages
    left_score = np.mean(left) * 100
    center_score = np.mean(center) * 100
    right_score = np.mean(right) * 100

    # Center-priority navigation heuristic
    center_threshold = 60

    if center_score >= center_threshold:
        decision = "FORWARD"

    elif left_score > right_score and left_score >= 60:
        decision = "LEFT"

    elif right_score > left_score and right_score >= 60:
        decision = "RIGHT"

    else:
        decision = "STOP"

    return (
        left_score,
        center_score,
        right_score,
        decision
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    image_path = os.path.join(
        BASE_DIR,
        "Data",
        "Data",
        "left25057.jpg"
    )

    # Predict traversability
    image, prediction_mask = predict_traversability(
        image_path
    )

    # Calculate navigation
    left, center, right, decision = navigation_decision(
        prediction_mask
    )

    # Display results
    print("Navigation Results")
    print("------------------")
    print(f"Left:     {left:.2f}%")
    print(f"Center:   {center:.2f}%")
    print(f"Right:    {right:.2f}%")
    print(f"Decision: {decision}")