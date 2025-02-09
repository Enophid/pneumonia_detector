from ultralytics import YOLO
import multiprocessing

def main():
    # Load a COCO-pretrained YOLOv8 model
    model = YOLO("yolo11s.pt")

    # Train the model
    results = model.train(
        data="dataset-detect/data.yaml",
        epochs=300,
        imgsz=640,
    )

if __name__ == '__main__':
    # Windows requires using spawn method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
