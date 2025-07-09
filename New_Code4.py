import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

time_now_1 = datetime.now().strftime("%H:%M:%S")
print("Time when the image is given to test:", time_now_1)

# Load and preprocess the image you want to detect objects in
image_path = 'D:/EDI_code/Images/dataset 2/test/class_1(pedestrain)/traffic9.webp'
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)

# Define the indices of your classes of interest
classes_of_interest = [0, 1, 2, 3, 4]  # Replace with your class indices

# Get bounding boxes, labels, and scores
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Set a score threshold for detected objects
score_threshold = 0.7

# Create a PIL image to draw bounding boxes
draw = ImageDraw.Draw(image)

# Iterate through the detected objects
for box, label, score in zip(boxes, labels, scores):
    if score > score_threshold and label.item() in classes_of_interest:
        box = [round(i, 5) for i in box.tolist()]  # Round box coordinates
        draw.rectangle(box, outline="blue", width=3)  # Draw bounding box
        label_text = f"Class {label.item()}"
        draw.text((box[0], box[1]), label_text, fill="red")

time_now_2 = datetime.now().strftime("%H:%M:%S")
print("Expected output's time is", time_now_2)

# Save or display the image with bounding boxes
image.show()

time1 = datetime.strptime(time_now_1, "%H:%M:%S")
time2 = datetime.strptime(time_now_2, "%H:%M:%S")

# Calculate the time difference
time_difference = time2 - time1

# Extract the time difference in seconds
seconds_difference = time_difference.total_seconds()

# Print the time difference
print("Time Difference:", time_difference)
print("Time Difference in seconds:", seconds_difference)



# import cap
# import label
# import torch
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.transforms import functional as F
# from PIL import Image, ImageDraw
# from datetime import datetime
# import matplotlib.pyplot as plt
# import cv2
#
# # Define the device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Load the pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model = model.to(device)
# model.eval()  # Set the model to evaluation mode
#
# time_now_1 = datetime.now().strftime("%H:%M:%S")
# print("Time when the video is given to test :", time_now_1)
#
# # Load and preprocess the video
# video_path = 'D:/EDI_code/Images/dataset 2/train/video/vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1793410.mov'
# cap = cv2.VideoCapture(video_path)
#
# # Create a VideoWriter object to save the output video
# output_path = 'output_video.mp4'
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#
# # Iterate through the frames of the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert the frame to PIL Image and preprocess
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
#     image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
#
#     # Perform inference
#     with torch.no_grad():
#         prediction = model(image_tensor)
#
#     # Define the indices of your classes of interest
#     classes_of_interest = [0, 1, 2, 3, 4]  # Replace with your class indices
#
#     # Get bounding boxes, labels, and scores
#     boxes = prediction[0]['boxes']
#     labels = prediction[0]['labels']
#     scores = prediction[0]['scores']
#
#     # Set a score threshold for detected objects
#     score_threshold = 0.7
#
#     # Draw bounding boxes on the frame
#     for box, label, score in zip(boxes, labels, scores):
#         if score > score_threshold and label.item() in classes_of_interest:
#             box = [round(i, 5) for i in box.tolist()]  # Round box coordinates
#             cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Draw bounding box
#             label_text = f"Class {label.item()}"
#             cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Write the frame with bounding boxes to the output video
#     out.write(frame)
#
#     # Display the frame with bounding boxes
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture and video writer
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# time_now_2 = datetime.now().strftime("%H:%M:%S")
# print("Expected output's time is", time_now_2)
#
# time1 = datetime.strptime(time_now_1, "%H:%M:%S")
# time2 = datetime.strptime(time_now_2, "%H:%M:%S")
#
# # Calculate the time difference
# time_difference = time2 - time1
#
# # Extract the time difference in seconds
# seconds_difference = time_difference.total_seconds()
#
# # Print the time difference
# print("Time Difference:", time_difference)
# print("Time Difference in seconds:", seconds_difference)


# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.transforms import functional as F
# from PIL import Image
# import cv2
# from datetime import datetime
#
# # Define the device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Load the pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model = model.to(device)
# model.eval()  # Set the model to evaluation mode
#
# # Open the video file for reading
# video_path = 'your_video_path.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Check if the video file is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()
#
# # Create the VideoWriter object to save the output video
# output_path = 'output_video.mp4'
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#
# # Set a score threshold for detected objects
# score_threshold = 0.7
#
# # Skip frames if needed (adjust this based on your real-time requirements)
# frame_skip = 2
# frame_count = 0
#
# batch_size = 4  # Adjust the batch size based on your hardware capabilities
#
# time_now_1 = datetime.now()
# print("Time when the video is given to test:", time_now_1)
#
# frame_buffer = []  # Buffer for frames to be processed in a batch
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_count += 1
#
#     # Skip frames based on frame_skip value
#     if frame_count % frame_skip != 0:
#         continue
#
#     # Convert the frame to PIL Image and preprocess
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
#     image_tensor = F.to_tensor(image).to(device)
#
#     # Append the frame to the buffer
#     frame_buffer.append(image_tensor)
#
#     # Process frames in batches
#     if len(frame_buffer) >= batch_size:
#         with torch.no_grad():
#             predictions = model([frame.to(device) for frame in frame_buffer])
#
#         # Iterate through batched predictions
#         for i, prediction in enumerate(predictions):
#             boxes = prediction[0]['boxes']
#             labels = prediction[0]['labels']
#             scores = prediction[0]['scores']
#
#             # Draw bounding boxes on the frame
#             for box, label, score in zip(boxes, labels, scores):
#                 if score > score_threshold:
#                     box = [round(coord, 2) for coord in box.tolist()]
#                     cv2.rectangle(frame, (int(box[0]), int(box[1]), int(box[2]), int(box[3]), (255, 0, 0), 2))
#                     label_text = f"Class {label.item()}",
#                     cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                 (0, 0, 255), 2),
#
#                     # Write the frame with bounding boxes to the output video
#                     out.write(frame),
#
#                     # Clear the frame buffer
#                     frame_buffer.clear(),
#
#                     # Release the video capture and video writer
#                     cap.release(),
#                     out.release(),
#

