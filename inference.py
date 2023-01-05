import time
import cv2
import numpy as np
from openvino.runtime import Core


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the given mean and standard deviation
    for CityScapes models.
    """
    image = image.astype(np.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /= 255.0
    image -= mean
    image /= std
    return image


def nms(labels, bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(labels[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return list(zip(picked_label, picked_boxes, picked_score))


if __name__ == "__main__":
    IMAGE_WIDTH = 1622
    IMAGE_HEIGHT = 626
    image_filename = "model/10060.png"
    raw_image = cv2.imread(image_filename)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized_image = normalize(resized_image)

    # Convert the resized images to network input shape.
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)

    # Load the network to OpenVINO Runtime.
    ie = Core()
    ir_path = "model/FasterRCNN/FasterRCNN.xml"
    # Load the network in OpenVINO Runtime.
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers.
    # output_boxes_layer_ir = compiled_model_ir.output(2)

    # Run inference on the input image.
    start = time.time()
    out = list(compiled_model_ir([input_image]).values())
    boxes, labels, probs = out
    outputs = nms(labels, boxes, probs, threshold=0.5)
    # print(outputs)
    print(f"Inference Time: {time.time()-start}")
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    for (label, boxes, probs) in outputs:
        # Draw a bounding box based on position.
        # Parameters in the `rectangle` function are: image, start_point, end_point, color, thickness.
        x_min, y_min, x_max, y_max = [int(coor) for coor in boxes]
        raw_image = cv2.rectangle(raw_image, (x_min, y_min), (x_max, y_max), colors["red"], 2)
        # Print the attributes of a vehicle.
        # Parameters in the `putText` function are: img, text, org, fontFace, fontScale, color, thickness, lineType.
        raw_image = cv2.putText(raw_image, f"{label}", (x_min, y_min - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["green"], 2, cv2.LINE_AA
        )
    cv2.imshow("Test", raw_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
