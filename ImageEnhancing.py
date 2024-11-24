import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Step 1: Load the image
image_path = 'images/14_ROI.jpg'
image = cv.imread(image_path)
assert image is not None, f"Could not load image from {image_path}"

# Step 2: Enhance sharpness using a kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
sharpened = cv.filter2D(image, -1, kernel)

# Step 3: Convert image to HSV for color filtering
hsv = cv.cvtColor(sharpened, cv.COLOR_BGR2HSV)

# Step 4: Define yellow color range in HSV
lower_yellow = np.array([20, 100, 100])  # Adjust these values as needed
upper_yellow = np.array([30, 255, 255])
mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Step 5: Apply morphological operations to remove noise
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # Remove small noise
mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_CLOSE, kernel)  # Close small gaps

# Step 6: Use Canny edge detection to enhance edges
edges = cv.Canny(mask_cleaned, threshold1=50, threshold2=150)

# Step 7: Combine the edge-detected regions with the mask
edges_dilated = cv.dilate(edges, kernel, iterations=1)  # Thicken edges
mask_refined = cv.bitwise_and(mask_cleaned, edges_dilated)

# Step 8: Apply the refined mask to the original sharpened image
yellow_regions = cv.bitwise_and(sharpened, sharpened, mask=mask_refined)

# Save and display intermediate steps for debugging
cv.imwrite('images/14_ROI_sharpened.jpg', sharpened)
cv.imwrite('images/14_ROI_yellow_mask.jpg', mask)
cv.imwrite('images/14_ROI_cleaned_mask.jpg', mask_cleaned)
cv.imwrite('images/14_ROI_edges.jpg', edges)
cv.imwrite('images/14_ROI_yellow_filtered.jpg', yellow_regions)

print("Preprocessing complete. Yellow regions isolated and saved.")

# Step 9: Load YOLO model and perform detection on the refined image
model = YOLO('yolov8n.pt')  # Replace with your trained model
results = model.predict(source=yellow_regions, conf=0.5)

# Step 10: Annotate YOLO results
annotated_image = yellow_regions.copy()
for result in results:
    if result.boxes:  # Ensure there are detections
        for box in result.boxes.xyxy:  # xyxy format: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box)  # Convert to integers
            cv.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Save and display the final annotated image
output_path = 'images/14_ROI_annotated_refined.jpg'
cv.imwrite(output_path, annotated_image)
print(f"Detection completed. Annotated image saved at {output_path}")

cv.imshow("Annotated Image", annotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
