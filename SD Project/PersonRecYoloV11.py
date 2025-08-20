import numpy as np
import json
import cv2 as cv  # Include OpenCV for visualization

# Post-processor class, must have fixed name 'PostProcessor'
class PostProcessor:
    def __init__(self, json_config):
        """
        Initialize the post-processor with configuration settings.

        Parameters:
            json_config (str): JSON string containing post-processing configuration.
        """
        self._json_config = json.loads(json_config)
        self._num_classes = int(self._json_config["POST_PROCESS"][0]["OutputNumClasses"])
        self._label_json_path = self._json_config["POST_PROCESS"][0]["LabelsPath"]
        self._input_height = int(self._json_config["PRE_PROCESS"][0]["InputH"])
        self._input_width = int(self._json_config["PRE_PROCESS"][0]["InputW"])
        self._output_conf_threshold = float(self._json_config["POST_PROCESS"][0].get("OutputConfThreshold", 0.5))  # Default threshold 0.5

        # Load label dictionary from JSON file
        with open(self._label_json_path, "r") as json_file:
            self._label_dictionary = json.load(json_file)

        # Set a filter for specific classes (e.g., "person")
        self._class_filter = ["person"]

    def forward(self, tensor_list, details_list):
        """
        Process the raw output tensor and visualize results using OpenCV.

        Parameters:
            tensor_list (list): List of tensors from the model.
            details_list (list): Additional details (unused in this example).

        Returns:
            list: A list of dictionaries containing detection results.
        """
        # Initialize results list
        new_inference_results = []

        # Extract and reshape the raw output tensor
        output_array = tensor_list[0].reshape(-1)

        # Index to parse the array
        index = 0
        for class_id in range(self._num_classes):
            num_detections = int(output_array[index])
            index += 1

            if num_detections == 0:
                continue

            for _ in range(num_detections):
                if index + 5 > len(output_array):
                    break

                score = float(output_array[index + 4])
                y_min, x_min, y_max, x_max = map(float, output_array[index:index + 4])
                index += 5

                if score < self._output_conf_threshold:
                    continue

                x_min = int(x_min * self._input_width)
                y_min = int(y_min * self._input_height)
                x_max = int(x_max * self._input_width)
                y_max = int(y_max * self._input_height)

                label = self._label_dictionary.get(str(class_id), f"class_{class_id}")

                # Filter detections for specific classes (e.g., "person")
                if label in self._class_filter:
                    result = {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "score": score,
                        "category_id": class_id,
                        "label": label,
                    }
                    new_inference_results.append(result)

        return new_inference_results

    def visualize(self, frame, results):
        """
        Visualize detection results on the given video frame.

        Parameters:
            frame (numpy.ndarray): The video frame.
            results (list): List of detection results.
        """
        person_cnt = 0

        for obj in results:
            x_min, y_min, x_max, y_max = obj["bbox"]
            label = obj["label"]
            score = obj["score"]

            if label == "person":  # Filter for "person" detections
                person_cnt += 1
                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Draw bounding box
                cv.putText(frame, f'{label}, Conf: {score:.2f}', (x_min, y_min - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display total count of detected people
        cv.putText(frame, f'Total Count: {person_cnt}', (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame
        cv.imshow('Person Detection', frame)