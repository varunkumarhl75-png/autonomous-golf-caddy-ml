import os
import matplotlib.pyplot as plt

from predict import predict_traversability, navigation_decision


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# Test images
test_images = [
    "left25057.jpg",
    "left25343.jpg"
]


for image_name in test_images:

    image_path = os.path.join(
        BASE_DIR,
        "Data",
        "Data",
        image_name
    )

    # Predict traversability
    image, prediction_mask = predict_traversability(
        image_path
    )

    # Get navigation decision
    left, center, right, decision = navigation_decision(
        prediction_mask
    )

    print("\n================================")
    print("Image:", image_name)
    print("================================")
    print(f"Left:     {left:.2f}%")
    print(f"Center:   {center:.2f}%")
    print(f"Right:    {right:.2f}%")
    print(f"Decision: {decision}")

    # Visualization
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    # Prediction mask
    plt.subplot(1, 2, 2)
    plt.imshow(prediction_mask, cmap="gray")
    plt.title(
        f"Traversability Prediction\n"
        f"Decision: {decision}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()