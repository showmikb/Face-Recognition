# Face-Recognition

To use the code, create two folders known_faces and unknown_faces, these are referenced in the code, so depending upon where you keep them you may have to change the path. 

Run the learn_faces.py This will read all the faces in the known_faces directory and attach a name to each encoded face.

Next, if you run the fc_recog.py file, it'll search through the unknown_faces directory and match them with the known faces.
If you run the fc_recog_live_video.py, it'll detect and recognize faces in real time. 
