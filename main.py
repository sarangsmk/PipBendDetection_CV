import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Step 1: Load and preprocess the image
image_path = 'images/14_ROI.jpg'
image = cv.imread(image_path)
assert image is not None, f"Could not load image from {image_path}"

# Enhance sharpness using a kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
sharpened = cv.filter2D(image, -1, kernel)

# Convert image to HSV for color filtering
hsv = cv.cvtColor(sharpened, cv.COLOR_BGR2HSV)

# Define yellow color range in HSV
lower_yellow = np.array([20, 100, 100])  # Adjust these values if needed
upper_yellow = np.array([30, 255, 255])
mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Apply the mask to isolate yellow areas
yellow_regions = cv.bitwise_and(sharpened, sharpened, mask=mask)

# Save the filtered image for debugging
cv.imwrite('images/14_ROI_yellow_filtered.jpg', yellow_regions)

# Step 2: Detect objects using YOLO
model = YOLO('yolov8n.pt')  # Replace with your trained model if available
results = model.predict(source=yellow_regions, conf=0.5)

# Extract bounding boxes (x_min, y_min, x_max, y_max) from results
all_widths = []
filtered_boxes = []
for result in results:
    for box in result.boxes.xyxy:  # YOLO results in xyxy format
        x_min, y_min, x_max, y_max = map(int, box[:4])
        width = x_max - x_min
        all_widths.append(width)

# Step 3: Calculate average width
if all_widths:
    average_width = sum(all_widths) / len(all_widths)
    print(f"Average Width of Pins: {average_width}")

    # Step 4: Filter detections based on width
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            width = x_max - x_min

            if width >= average_width:  # Retain only boxes with width >= average
                filtered_boxes.append((x_min, y_min, x_max, y_max))

    # Annotate the filtered results on the image
    annotated_image = image.copy()
    for (x_min, y_min, x_max, y_max) in filtered_boxes:
        cv.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Save the annotated image
    output_path = 'images/14_ROI_filtered.jpg'
    cv.imwrite(output_path, annotated_image)
    print(f"Filtered detection saved at {output_path}")

    # Optionally display the result
    cv.imshow("Filtered Detection", annotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No pins detected.")
