import face_recognition
import cv2 as cv2 #OpenCV
facecascade = cv2.CascadeClassifier('C:/Users/engel/Downloads/haarcascade_frontalface_default.xml')
import os

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "C:/Users/engel/Downloads/employee"

known_face_encodings =[]
known_face_names =[]

for file in os.listdir(employee_pictures):
    employee, extension = file.split(".")
    img = face_recognition.load_image_file(employee_pictures+"/"+file)
    face_location = face_recognition.face_locations(img ,number_of_times_to_upsample=0)
    img_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(img_encoding)
    known_face_names.append(employee)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
print('employee representations retrieved successfully')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame ,number_of_times_to_upsample=0)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.6)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
           

            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('frame',frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
