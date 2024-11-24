import cv2 as cv
import numpy as np
from ultralytics import YOLO
import datetime

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
lower_yellow = np.array([20, 100, 100])  # Adjust these values as needed
upper_yellow = np.array([30, 255, 255])
mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Apply the mask to isolate yellow areas
yellow_regions = cv.bitwise_and(sharpened, sharpened, mask=mask)

# Save the processed image for reference
cv.imwrite('images/14_ROI_yellow_filtered.jpg', yellow_regions)
print("Yellow regions isolated and saved.")

# Step 2: Load YOLO model
model = YOLO('yolov8n.pt')  # Replace with your trained model once available

# Step 3: Perform detection (use the yellow-filtered image)
results = model.predict(source=yellow_regions, conf=0.5)  # Detect objects in yellow-filtered regions

# Initialize variables for width measurement
total_width = 0
region_count = 0

# Step 4: Process YOLO results and calculate widths
annotated_image = yellow_regions.copy()
for result in results:
    if result.boxes:  # Ensure there are detections
        for box in result.boxes.xyxy:  # xyxy format: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box)  # Convert to integers
            width = xmax - xmin
            total_width += width
            region_count += 1
            # Draw YOLO bounding boxes
            cv.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Calculate and display average width if YOLO detects pins
if region_count > 0:
    average_width = total_width / region_count
    print(f"Average width of detected yellow regions: {average_width:.2f} pixels")
else:
    print("No YOLO detections found. Average width cannot be calculated.")

# Step 5: Find contours in the yellow mask for further processing
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#Calculate the average width
average_width_detected = 0
width_sum = 0
width_count = 0
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)  # Get bounding box of the contour
    print(f"Width -{w}")
    width_sum += w
    width_count +=1

#Average width calculation
average_width_detected = width_sum/width_count

#print width
print(f"Total available widths -{width_count}")
print(f"Width Sum -{width_sum}")
print(f"Average Width -{average_width_detected}")

# Remove noise
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)  # Get bounding box of the contour
    # Draw contour-based bounding boxes if the item's width is greater than the average detected width
    if w <= average_width_detected:
        cv.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED) # remove these detections/items whose width is lesser - fill black

# Detect bend - sets which have higher width than the average width.
for contour in contours:
    x, y, w, h = cv.boundingRect( )  # Get bounding box of the contour
    # Draw contour-based bounding boxes if the item's width is greater than the average detected width
    if w > average_width_detected and w > 50:
        cv.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save and display the final annotated image
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = f'images/14_ROI_annotated_combined_{timestamp}.jpg'
cv.imwrite(output_path, annotated_image)
print(f"Combined annotated image saved at {output_path}")

cv.imshow("Annotated Image", annotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
