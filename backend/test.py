from ultralytics import YOLO
import cv2
from pathlib import Path
import time

def test_model():
    # Initialize paths
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "dataset-detect" / "test" / "images"
    output_dir = project_root / "test_results" / "positive_cases"
    model_path = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Initialize counters
    total_images = 0
    positive_cases = 0
    
    print(f"Processing images from {test_dir}")
    
    # Process each image in test directory
    for img_path in test_dir.glob("*.jpg"):
        total_images += 1
        print(f"Processing {img_path.name}...")
        
        # Read image
        img = cv2.imread(str(img_path))
        
        # Run inference
        results = model.predict(img, conf=0.25)
        result = results[0]
        
        # Check if pneumonia detected
        if len(result.boxes) > 0:
            positive_cases += 1
            
            # Draw boxes on positive cases
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Draw rectangle
                cv2.rectangle(img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Add confidence text
                cv2.putText(img, 
                           f'Pneumonia: {confidence:.2f}', 
                           (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            # Save image with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = output_dir / f"pneumonia_{confidence:.2f}_{timestamp}_{img_path.name}"
            cv2.imwrite(str(output_path), img)
            print(f"Saved positive case: {output_path.name}")
    
    # Print summary
    print("\nTest Results:")
    print(f"Total images processed: {total_images}")
    print(f"Positive cases found: {positive_cases}")
    print(f"Detection rate: {(positive_cases/total_images)*100:.2f}%")
    print(f"\nPositive cases saved in: {output_dir}")

if __name__ == "__main__":
    test_model()