# Standard library imports
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf

# Configure TensorFlow for GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) found and configured: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Import MTCNN and FaceNet after configuring GPU
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# Local imports
from cv2_enumerate_cameras import enumerate_cameras

# State constants
STATE_ALIGNING = 0
STATE_COUNTDOWN = 1
STATE_CAPTURED = 2


def init_models():
    global detector, facenet
    # Initialize models with GPU support
    detector = MTCNN(
        stages="face_and_landmarks_detection",  # Use valid stage
        device="gpu:0",  # Specify GPU device
    )
    facenet = FaceNet()
    # Warm up GPU with a dummy inference
    dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
    facenet.embeddings([dummy_image])


def process_face(args):
    """Helper function to process a single face using FaceNet"""
    image_path, name = args
    global detector, facenet
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face using MTCNN
        faces = detector.detect_faces(image)
        if faces:
            x, y, w, h = faces[0]["box"]
            face = image[y : y + h, x : x + w]
            face = cv2.resize(face, (160, 160))

            # Get face embedding
            face_embedding = facenet.embeddings([face])[0]
            return (face_embedding, name)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return None


def load_dataset():
    """Load face dataset and compute FaceNet embeddings"""
    cache_file = "facenet_embeddings_cache.pkl"
    global detector

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


def get_face_angle(face):
    """Calculate lateral face angle (yaw) using MTCNN landmarks
    Returns angle in degrees where:
    0 degrees = facing camera directly
    Positive = face turned to their left (our right)
    Negative = face turned to their right (our left)
    """
    landmarks = face["keypoints"]
    nose = landmarks["nose"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    # Calculate eye midpoint
    eye_midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Calculate distance between eyes
    eye_distance = np.sqrt(
        (right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2
    )

    # Calculate horizontal distance from nose to eye midpoint
    nose_offset = nose[0] - eye_midpoint[0]

    # Convert to angle using arctangent
    # Multiply by 2 to amplify the angle for better sensitivity
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


def get_alignment_instruction(angle):
    if abs(angle) < 8:  # Wider threshold
        return "Face aligned properly", True
    elif angle < -8:
        return "Turn face right slowly", False
    else:
        return "Turn face left slowly", False


class FaceVerificationState:
    def __init__(self):
        self.current_state = STATE_ALIGNING
        self.angle_buffer = AngleBuffer(10)
        self.stable_frames = 0
        self.state_start_time = 0
        self.captured_frame = None
        self.captured_result = ""
        self.captured_face_loc = None
        self.captured_box_color = None

    def reset(self):
        self.__init__()


def main():
    init_models()
    global detector
    for camera_info in enumerate_cameras():
        print(f"{camera_info.index}: {camera_info.name}")
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)  # Width of the frames in the video stream.
    capture.set(4, 720)  # Height of the frames in the video stream.

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

        # Process face detection using MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        alignment_msg = "No face detected"
        result_msg = state.captured_result
        box_color = (0, 0, 255)

        if faces:
            face = faces[0]  # Get the first detected face
            angle = get_face_angle(face)
            state.angle_buffer.add(angle)
            smoothed_angle = state.angle_buffer.get_average()
            alignment_msg, is_aligned = get_alignment_instruction(smoothed_angle)

            if state.current_state == STATE_CAPTURED:
                if is_aligned:
                    # Get face embedding using FaceNet
                    x, y, w, h = face["box"]
                    face_img = rgb_frame[y : y + h, x : x + w]
                    face_img = cv2.resize(face_img, (160, 160))
                    face_embedding = facenet.embeddings([face_img])[0]

                    # Calculate similarities with known faces
                    face_distances = np.linalg.norm(
                        known_face_encodings - face_embedding, axis=1
                    )
                    best_match_index = np.argmin(face_distances)
                    match_percentage = (1 - face_distances[best_match_index]) * 100

                    if match_percentage >= 0:  # Adjusted threshold for FaceNet
                        name = known_face_names[best_match_index]
                        result_msg = f"Match found: {name} ({match_percentage:.2f}%)"
                    else:
                        result_msg = f"No match found ({match_percentage:.2f}%)"

                    state.captured_frame = frame.copy()
                    state.captured_result = result_msg
                    state.captured_face_loc = (y, x + w, y + h, x)
                    state.captured_box_color = (255, 0, 0)

            # Rest of the state handling logic remains the same
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

            # Draw face boxes using MTCNN coordinates
            x, y, w, h = face["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Only draw UI elements if not in CAPTURED state
        if state.current_state != STATE_CAPTURED:
            cv2.putText(
                frame,
                alignment_msg,
                (10, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if result_msg:
                cv2.putText(
                    frame,
                    result_msg,
                    (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

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
