import torch
import cv2
import numpy as np

model_type = "MiDaS_small" 

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_vis = (depth_map - depth_min) / (depth_max - depth_min)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    # Apply colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Show the original frame and depth map side by side
    combined = np.hstack((frame, depth_colormap))
    cv2.imshow('Webcam Depth Estimation (Original | Depth)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

