import face_recognition
import os
import pickle

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
known_faces = list()
known_names = list()


def detect_known_faces():
    print('Loading known faces...')

    for name in os.listdir(KNOWN_FACES_DIR):

        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

            if face_recognition.face_encodings(image):
                encoding = face_recognition.face_encodings(image)[0]

            else:
                print("Skipping", filename)
                continue
            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)

    with open('knownFaces', 'wb') as fp1:
        pickle.dump(known_faces, fp1)
    with open('knownNames', 'wb') as fp2:
        pickle.dump(known_names, fp2)


if __name__ == '__main__':
    detect_known_faces()