import cv2
import face_recognition


#load the sample image and get the 128 face embeddings that is vecotrs from them
praj_image= face_recognition.load_image_file('1.jpg')
modi_image= face_recognition.load_image_file('modi2.jpg')
trump_image= face_recognition.load_image_file('trump.jpg')

#here we are assuming that the image is having only a single face
face_encodings_praj = face_recognition.face_encodings(praj_image)[0]
face_encodings_modi = face_recognition.face_encodings(modi_image)[0]
face_encodings_trump = face_recognition.face_encodings(trump_image)[0]

known_face_encodings = [face_encodings_praj,face_encodings_modi, face_encodings_trump]
known_face_names = ["Prajwal", "Narendra Modi", "Donald Trump"]

#load the unknown image to recognise faces in it
img_to_detect = cv2.imread('both3.jpg')
original_img = img_to_detect

#detect all faces in the image
all_face_locations = face_recognition.face_locations(img_to_detect, model = 'cnn', number_of_times_to_upsample=1)

#detect all face embeddings for all faces detected
all_face_encodings = face_recognition.face_encodings(img_to_detect,all_face_locations)

#print the number of faces detected
print(f"There are {len(all_face_locations)} no of faces in this image")

for current_face_location, current_face_encoding in zip(all_face_locations,all_face_encodings):
    (top,right,bottom, left) = current_face_location
    # print(f"Found face {index} at top:{top},right:{right}, bottom:{bottom},left:{left}")

    all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
    print(all_matches)

    #string to hold the label
    name_of_person = 'Unknown_face'
    #Check if the all_matches at least one time
    #if yes get the index number of the face that is in the first index of all matches
    if True in all_matches:
        first_match_index = all_matches.index(True)
        print(first_match_index)
        name_of_person = known_face_names[first_match_index]
    #Draw rectangle around the code
    cv2.rectangle(original_img,(left,top), (right,bottom),(255,0,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_img,name_of_person, (left,bottom), font, 0.5, (255,255,0),1 )

    # current_face_image = image[top:bottom,left:right]
    cv2.imshow('Face no',original_img)
    cv2.waitKey(0)