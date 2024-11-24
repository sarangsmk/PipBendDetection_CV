import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('model/yolov11n-100-v1.pt')

# Load the local image
image_path = "images/14_ROI.jpg"  # Replace with your image path
rgb = cv2.imread(image_path)

if rgb is None:
    raise Exception(f"Failed to load image from {image_path}")

# Process the captured image
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gaus_blur = cv2.GaussianBlur(gray, (5, 7), sigmaX=3, sigmaY=5)

# Set up SimpleBlobDetector parameters for pin counting
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255  # Detect light blobs
params.minThreshold = 40
params.maxThreshold = 255
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1.0
params.filterByConvexity = False
params.filterByInertia = False

# Create a SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the blurred grayscale image
keypoints = detector.detect(gaus_blur)

# Draw detected blobs as red circles on the original image
blob_image = cv2.drawKeypoints(
    rgb, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display pin count as text on the image
pin_count_text = f"Pin count: {len(keypoints)}"
cv2.putText(blob_image, pin_count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 255), 2, cv2.LINE_AA)

# Run YOLO model to detect objects in the original image
results = model(rgb)

# Initialize an empty set for unique labels
labels = set()

# Extract label names from detections
for box in results[0].boxes:
    # Get the label name for each detected object
    label = box.cls
    labels.add(model.names[int(label)])  # Convert class index to label name using model.names

# Draw detected label names on the image
y_position = 100
for label in labels:
    cv2.putText(blob_image, label, (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)
    y_position += 30  # Increment position for the next label

# Display the processed image with pin count and detected labels
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processed Image", 1280, 960)
cv2.imshow("Processed Image", blob_image)

# Wait and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Output the number of detected pins and labels in the console
print(f"Number of detected pins: {len(keypoints)}")
print(f"Detected labels: {list(labels)}")
