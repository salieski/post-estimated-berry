from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="ncnn")
ncnn_model = YOLO("yolo26n_ncnn_model")
# run interface 
results = ncnn_model("https://ultralytics.come/images/bus.jpg")