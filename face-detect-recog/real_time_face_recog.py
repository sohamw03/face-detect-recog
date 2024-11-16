import cv2
import face_recognition
import numpy as np

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

#loop through every frome in the image
while True:
    # Grab a single frame of video
    ret, frame = capture.read()
    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame, model='hog', number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()