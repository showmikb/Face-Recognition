import face_recognition
import os
import cv2
import numpy as np
import pickle

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'sift'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

class FaceLiveRecognition:
    def __init__(self):
        known_faces=[]
        known_names=[]


    def learn_from_pickle(self):
        with open('knownFaces', 'rb') as fp1:
            self.known_faces = pickle.load(fp1)
        with open('knownNames', 'rb') as fp2:
            self.known_names = pickle.load(fp2)


    # Returns (R, G, B) from name
    def name_to_color(self, name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color





    def recognize_face(self, filename, imag):
        print('Trying to recognize faces...')
        image = imag
        locations = face_recognition.face_locations(image, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(self.known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = self.known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = self.name_to_color(match)

                # Paint frame
                # cv2.rectangle(imag, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame for name
                cv2.rectangle(imag, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(imag, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), FONT_THICKNESS)

        # Show image
        cv2.imshow(filename, imag)


    def detect_unknown_faces_from_video(self):
        print('Loading video ...')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            self.recognize_face('image',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fc_recog = FaceLiveRecognition()
    fc_recog.learn_from_pickle()
    fc_recog.detect_unknown_faces_from_video()
