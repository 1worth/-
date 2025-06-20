from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

results = model("C:/Users/Lenovo/Desktop/0082.jpg",conf=0.25,save=True)
