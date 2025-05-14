from ultralytics import YOLO

if __name__ == '__main__':
	model = YOLO("./runs/obb/train/weights/last.pt")  # build from YAML and transfer weights
	validation_results = model.val(data="etdii.yaml")