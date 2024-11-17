import cv2
import face_recognition
import numpy as np
import time
from cv2_enumerate_cameras import enumerate_cameras

def get_face_angle(face_landmarks):
    # Calculate angle using eye positions instead of nose-chin
    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return eye_angle

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
    if abs(angle) < 3:  # Wider threshold
        return "Face aligned properly", True
    elif angle < -3:
        return "Turn face right slowly", False
    else:
        return "Turn face left slowly", False


for camera_info in enumerate_cameras():
    print(f'{camera_info.index}: {camera_info.name}')
capture = cv2.VideoCapture(0)
capture.set(3, 1280)  # Width of the frames in the video stream.
capture.set(4, 720)  # Height of the frames in the video stream.

# load the sample image and get the 128 face embeddings that is vecotrs from them
soham_image= face_recognition.load_image_file('soham.jpg')
modi_image= face_recognition.load_image_file('modi2.jpg')
trump_image= face_recognition.load_image_file('trump.jpg')

# here we are assuming that the image is having only a single face
face_encodings_soham = face_recognition.face_encodings(soham_image)[0]
face_encodings_modi = face_recognition.face_encodings(modi_image)[0]
face_encodings_trump = face_recognition.face_encodings(trump_image)[0]

known_face_encodings = [face_encodings_soham,face_encodings_modi, face_encodings_trump]
known_face_names = ["Soham", "Narendra Modi", "Donald Trump"]

process_this_frame = True

# Add UI window
cv2.namedWindow('Face Verification System')
font = cv2.FONT_HERSHEY_DUPLEX

# State constants
STATE_ALIGNING = 0
STATE_COUNTDOWN = 1
STATE_CAPTURED = 2

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

# Initialize state
state = FaceVerificationState()

# Constants
REQUIRED_STABLE_FRAMES = 15
ALIGNMENT_DELAY = 1
COUNTDOWN_DURATION = 3

#loop through every frome in the image
while True:
    # Grab a single frame of video
    ret, frame = capture.read()
    current_time = time.time()

    # Show frozen frame if in CAPTURED state
    if state.current_state == STATE_CAPTURED and state.captured_frame is not None:
        frame = state.captured_frame.copy()
        cv2.putText(frame, state.captured_result, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        if state.captured_face_loc:
            top, right, bottom, left = state.captured_face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), state.captured_box_color, 2)
        cv2.imshow('Face Verification System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state.reset()
        continue

    # Process face detection every frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)

    # Initialize display variables
    alignment_msg = "No face detected"
    result_msg = state.captured_result
    box_color = (0, 0, 255)  # Default red

    if face_locations:
        face_landmarks_list = face_recognition.face_landmarks(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        if face_encodings:
            angle = get_face_angle(face_landmarks_list[0])
            state.angle_buffer.add(angle)
            smoothed_angle = state.angle_buffer.get_average()
            alignment_msg, is_aligned = get_alignment_instruction(smoothed_angle)

            if state.current_state == STATE_ALIGNING:
                if is_aligned:
                    state.stable_frames += 1
                    progress = min(state.stable_frames / REQUIRED_STABLE_FRAMES * 100, 100)
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
                    remaining = COUNTDOWN_DURATION - int(current_time - state.state_start_time)
                    if remaining > 0:
                        alignment_msg = f"Keep still: {remaining}..."
                        box_color = (0, 255, 255)
                    else:
                        state.current_state = STATE_CAPTURED

            elif state.current_state == STATE_CAPTURED:
                if not is_aligned:
                    state.reset()
                    alignment_msg = "Face misaligned, please try again"
                else:
                    # Process face recognition
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                    best_match_index = np.argmin(face_distances)
                    match_percentage = (1 - face_distances[best_match_index]) * 100

                    if match_percentage >= 50:
                        name = known_face_names[best_match_index]
                        result_msg = f"Match found: {name} ({match_percentage:.1f}%)"
                    else:
                        result_msg = f"No match found ({match_percentage:.1f}%)"

                    state.captured_frame = frame.copy()
                    state.captured_result = result_msg
                    state.captured_face_loc = (top * 4, right * 4, bottom * 4, left * 4)
                    state.captured_box_color = (255, 0, 0)

    # Only draw UI elements if not in CAPTURED state
    if state.current_state != STATE_CAPTURED:
        cv2.putText(frame, alignment_msg, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        if result_msg:
            cv2.putText(frame, result_msg, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        # Draw face boxes
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        cv2.imshow('Face Verification System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        state.reset()

capture.release()
cv2.destroyAllWindows()