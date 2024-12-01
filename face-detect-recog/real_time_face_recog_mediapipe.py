import os, sys, time, tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import supervision as sv
import mediapipe as mp
from cv2_enumerate_cameras import enumerate_cameras
from dotenv import load_dotenv
from inference import get_model

load_dotenv()

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) found and configured: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

from keras_facenet import FaceNet
import pickle

# State constants
STATE_ALIGNING = 0
STATE_COUNTDOWN = 1
STATE_CAPTURED = 2


def init_models():
    global face_detector, facenet, accessory_model, label_annotator, box_annotator
    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5  # 1 for full range detection
    )

    facenet = FaceNet()
    # Warm up GPU with a dummy inference
    dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
    facenet.embeddings([dummy_image])

    # Initialize accessory model using get_model
    accessory_model = get_model(model_id="nydata2/1", api_key=os.getenv("ROBOFLOW_API_KEY"))

    # Initialize annotators for visualization
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("Initialized accessory detection model")


def process_face(args):
    """Helper function to process a single face using MediaPipe and FaceNet"""
    image_path, name = args
    global face_detector, facenet
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face using MediaPipe
        results = face_detector.process(image_rgb)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
            w, h = int(bbox.width * iw), int(bbox.height * ih)

            # Extract and process face
            face = image_rgb[y : y + h, x : x + w]
            face = cv2.resize(face, (160, 160))
            face_embedding = facenet.embeddings([face])[0]
            return (face_embedding, name)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return None


def load_dataset():
    """Load face dataset and compute FaceNet embeddings"""
    cache_file = "facenet_embeddings_cache.pkl"

    # Try to load from cache first
    if os.path.exists(cache_file):
        print("Loading faces from cache...")
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                print(f"Loaded {len(cache_data['names'])} faces from cache")
                return cache_data["encodings"], cache_data["names"]
        except Exception as e:
            print(f"Cache load failed: {e}")

    # If cache doesn't exist or failed to load, process the dataset
    print("Processing face dataset...")
    # Read CSV file
    df = pd.read_csv("archive/Dataset.csv")

    known_face_encodings = []
    known_face_names = []
    processed_names = {}  # Track count of images per person
    tasks = []

    # Prepare tasks for parallel processing
    faces_dir = "archive/Faces/Faces"
    for _, row in df.iterrows():
        name = row["label"].split("_")[0]
        if processed_names.get(name, 0) >= 5:
            continue

        image_path = os.path.join(faces_dir, row["id"])
        tasks.append((image_path, name))
        processed_names[name] = processed_names.get(name, 0) + 1

    # Process faces in parallel
    with ProcessPoolExecutor(
        max_workers=int(os.cpu_count() * 2), initializer=init_models
    ) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(process_face, tasks),
                total=len(tasks),
                desc="Loading faces",
            )
        )

    # Filter out None results and separate encodings and names
    for result in results:
        if result:
            encoding, name = result
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    # Save to cache after processing
    try:
        cache_data = {"encodings": known_face_encodings, "names": known_face_names}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print("Saved face encodings to cache")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return known_face_encodings, known_face_names


def get_face_angle(detection, image_width, image_height):
    """Calculate face angle using MediaPipe landmarks"""
    # Get eye landmarks
    right_eye = detection.location_data.relative_keypoints[0]
    left_eye = detection.location_data.relative_keypoints[1]
    nose = detection.location_data.relative_keypoints[2]

    # Convert relative coordinates to absolute
    right_eye_x = int(right_eye.x * image_width)
    right_eye_y = int(right_eye.y * image_height)
    left_eye_x = int(left_eye.x * image_width)
    left_eye_y = int(left_eye.y * image_height)
    nose_x = int(nose.x * image_width)
    nose_y = int(nose.y * image_height)

    # Calculate eye midpoint
    eye_midpoint = ((left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2)

    # Calculate angle
    eye_distance = np.sqrt(
        (right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2
    )
    nose_offset = nose_x - eye_midpoint[0]
    angle = np.degrees(2 * np.arctan2(nose_offset, eye_distance))

    return angle


class AngleBuffer:
    def __init__(self, size=10):
        self.size = size
        self.angles = []

    def add(self, angle):
        self.angles.append(angle)
        if len(self.angles) > self.size:
            self.angles.pop(0)

    def get_average(self):
        if not self.angles:
            return 0
        return sum(self.angles) / len(self.angles)


class AccessoryBuffer:
    def __init__(self, size=10):
        self.size = size
        self.detections = []

    def add(self, has_forbidden):
        self.detections.append(has_forbidden)
        if len(self.detections) > self.size:
            self.detections.pop(0)

    def is_consistently_detected(self):
        if not self.detections:
            return False
        # Return True if accessories detected in majority of recent frames
        return sum(self.detections) / len(self.detections) == 0.8


def get_alignment_instruction(angle):
    threshold = 10
    if abs(angle) < threshold:
        return "Face aligned properly", True
    elif angle < -threshold:
        return "Turn face right slowly", False
    else:
        return "Turn face left slowly", False


class FaceVerificationState:
    def __init__(self):
        self.current_state = STATE_ALIGNING
        self.angle_buffer = AngleBuffer(20)
        self.accessory_buffer = AccessoryBuffer(60)
        self.stable_frames = 0
        self.state_start_time = 0
        self.captured_frame = None
        self.captured_result = ""
        self.accessory_msg = ""
        self.captured_face_loc = None
        self.captured_box_color = None

    def reset(self):
        self.__init__()


def process_accessories(frame):
    """Process frame for accessory detection using Supervision"""
    min_confidence = 0.6
    try:
        results = accessory_model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        forbidden_accessories = ["cap", "glasses", "hat", "sunglasses"]

        # Check if any forbidden accessories are detected
        labels = detections.data.get("class_name", [])
        confidences = detections.confidence
        label_dict = dict(zip(labels, confidences))
        has_forbidden = any(
            label in forbidden_accessories
            and label_dict.get(label, 0) >= min_confidence
            for label in labels
        )

        # Annotate frame with boxes and labels
        annotated_frame = frame.copy()
        if has_forbidden:
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        return annotated_frame, labels, has_forbidden
    except Exception as e:
        print(f"Accessory detection error: {e}")
        return frame, [], False


def process_frame(frame):
    """Process a frame using MediaPipe face detection"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)

    if not results.detections:
        return None, None, None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    face_img = frame_rgb[y : y + h, x : x + w]
    face_img = cv2.resize(face_img, (160, 160))

    return detection, face_img, (x, y, w, h)


def main():
    init_models()
    for camera_info in enumerate_cameras():
        print(f"{camera_info.index}: {camera_info.name}")
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)  # Width
    capture.set(4, 720)  # Height

    # Load dataset instead of individual images
    print("Loading face dataset...")
    known_face_encodings, known_face_names = load_dataset()
    print(f"Loaded {len(known_face_names)} unique faces")

    process_this_frame = True

    # Add UI window
    cv2.namedWindow("Face Verification System")
    font = cv2.FONT_HERSHEY_DUPLEX

    # Initialize state
    state = FaceVerificationState()

    # Constants
    REQUIRED_STABLE_FRAMES = 10
    ALIGNMENT_DELAY = 1
    COUNTDOWN_DURATION = 2.25
    COUNTDOWN_INTERVAL = 0.75

    # loop through every frome in the image
    while True:
        # Grab a single frame of video
        ret, frame = capture.read()
        current_time = time.time()
        frame = cv2.flip(frame, 1)

        # Show frozen frame if in CAPTURED state
        if state.current_state == STATE_CAPTURED and state.captured_frame is not None:
            frame = state.captured_frame.copy()
            cv2.putText(
                frame,
                state.captured_result,
                (10, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if state.captured_face_loc:
                top, right, bottom, left = state.captured_face_loc
                cv2.rectangle(
                    frame, (left, top), (right, bottom), state.captured_box_color, 2
                )
            cv2.imshow("Face Verification System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                state.reset()
            continue

        # Process frame with MediaPipe
        detection, face_img, face_bbox = process_frame(frame)

        alignment_msg = "No face detected"
        result_msg = state.captured_result
        accessory_msg = ""
        box_color = (0, 0, 255)

        if detection:
            # Get face angle using MediaPipe landmarks
            angle = get_face_angle(detection, frame.shape[1], frame.shape[0])
            state.angle_buffer.add(angle)
            smoothed_angle = state.angle_buffer.get_average()
            alignment_msg, is_aligned = get_alignment_instruction(smoothed_angle)

            if state.current_state == STATE_CAPTURED:
                if is_aligned:
                    # Check for accessories before proceeding with recognition
                    _, _, has_forbidden = process_accessories(frame)
                    if has_forbidden:
                        state.reset()
                        result_msg = ("Accessories detected. Please remove them and try again")
                        continue

                    # Get face embedding using FaceNet - using face_img from process_frame
                    face_embedding = facenet.embeddings([face_img])[0]

                    # Calculate similarities with known faces
                    face_distances = np.linalg.norm(known_face_encodings - face_embedding, axis=1)
                    best_match_index = np.argmin(face_distances)
                    match_percentage = (1 - face_distances[best_match_index]) * 100

                    if match_percentage >= 0:  # Adjusted similarity threshold for FaceNet
                        name = known_face_names[best_match_index]
                        result_msg = f"Match found: {name}"
                    else:
                        result_msg = f"No match found"

                    x, y, w, h = face_bbox  # Use returned bbox coordinates
                    state.captured_frame = frame.copy()
                    state.captured_result = result_msg
                    state.captured_face_loc = (y, x + w, y + h, x)
                    state.captured_box_color = (255, 0, 0)

            if state.current_state == STATE_ALIGNING:
                if is_aligned:
                    state.stable_frames += 1
                    progress = min(
                        state.stable_frames / REQUIRED_STABLE_FRAMES * 100, 100
                    )
                    alignment_msg += f" ({progress:.0f}%)"

                    if state.stable_frames >= REQUIRED_STABLE_FRAMES:
                        if state.state_start_time == 0:
                            state.state_start_time = current_time
                        elif current_time - state.state_start_time >= ALIGNMENT_DELAY:
                            state.current_state = STATE_COUNTDOWN
                            state.state_start_time = current_time
                else:
                    state.stable_frames = 0
                    state.state_start_time = 0

                box_color = (0, 255, 0) if is_aligned else (0, 0, 255)

            elif state.current_state == STATE_COUNTDOWN:
                if not is_aligned:
                    state.reset()
                    alignment_msg = "Please keep face aligned"
                else:
                    remaining_duration = COUNTDOWN_DURATION - (
                        current_time - state.state_start_time
                    )
                    remaining_count = min(
                        3, int(remaining_duration / COUNTDOWN_INTERVAL) + 1
                    )
                    if remaining_count > 0:
                        alignment_msg = f"Keep still: {remaining_count}..."
                        box_color = (0, 255, 255)
                    else:
                        state.current_state = STATE_CAPTURED

            # Draw face box using MediaPipe detection
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Process accessories if not in CAPTURED state
        if state.current_state != STATE_CAPTURED:
            frame, accessories, has_forbidden = process_accessories(frame)
            state.accessory_buffer.add(has_forbidden)

            # Update accessory message
            if (has_forbidden and state.current_state in [STATE_ALIGNING, STATE_COUNTDOWN] and state.accessory_buffer.is_consistently_detected()):
                state.reset()
                accessory_msg = "Please remove accessories"

        # Update UI elements
        if state.current_state != STATE_CAPTURED:
            cv2.putText(frame, alignment_msg, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2,)
            if result_msg:
                cv2.putText(frame, result_msg, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2,)
            if accessory_msg:
                cv2.putText(frame, accessory_msg, (10, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2,)

            cv2.imshow("Face Verification System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            state.reset()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
