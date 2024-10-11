# So this loads the openCV
import cv2
import numpy as np

print("OpenCL available:", cv2.ocl.haveOpenCL())  # Should return True
print("OpenCL in use:", cv2.ocl.useOpenCL())      # Should return True if OpenCL is enabled

cv2.ocl.setUseOpenCL(True)
#gotta load YOLO
def load_yolo():
    # Load the YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Load the COCO dataset class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get the layer names from the network
    layer_names = net.getLayerNames()
    
    # Get the output layer indices
    output_layer_indices = net.getUnconnectedOutLayers()
    
    # Fix for OpenCV's getUnconnectedOutLayers(), assuming it returns a 1D list of indices
    output_layers = [layer_names[i - 1] for i in output_layer_indices.flatten()]
    
    return net, classes, output_layers


#process frame and run detection
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.0392, (416, 416), (0, 0, 0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #Initialization
    class_ids = []
    confidences = []
    boxes = []

    #Process detection data
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Confidence threshold
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[0] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #Non-maximum supression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indices

# Draw bounding boxes on detected objects
def draw_labels(img, boxes, confidences, class_ids, classes, indices):
    if len(indices) == 0:
        print("No objects detected")
    else:
        print(f"Detected {len(indices)} objects")
    
    for i in indices:
        # Unpack index if necessary
        if isinstance(i, (list, tuple)):  
            i = i[0]  # Flatten the index if it is a list or tuple
        
        # Extract box coordinates and other details
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        
        # Draw the bounding box and label on the image
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def start_video_capture():
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0) # 0 for webcam (need one for Sony)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, confidences, class_ids, indices = detect_objects(frame, net, output_layers)
        draw_labels(frame, boxes, confidences, class_ids, classes, indices)

        cv2.imshow("Object Detection", frame)

        #press q to quit

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_capture()

