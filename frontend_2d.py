import requests
import cv2
import json

import time

from src.utils.pose import Pose2D
from src.utils.drawer import Drawer
from sys import argv


"""
Exemple of python frontend for backend.py

Run from terminal : python frontend_2d.py [movie_file_path] [port]
with all parameters optional
Keep holding the backspace key to speed the video 30x


To run on an external server :
1.- run the backend on the server
2.- redirect the backend selected port to your local computer using :
ssh -i ~/.ssh/server_private_key -f user@ip -L port_frontend:localhost:port_backend -N
3.- start the frontend on localhost:port_frontend
"""



def start(file_path, port):

    cap = cv2.VideoCapture(file_path)

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frameData = cv2.imencode('.jpg', frame)[1].tostring()

        tmpTime = time.time()
        r = requests.post("http://localhost:"+port, headers={'Content-Type': 'application/octet-stream'}, data=frameData)
        fps = int(1/(time.time()-tmpTime))

        jsonPose = json.loads(r.text)
        poses = [Pose2D.build_from_JSON(p['pose_2d']) for p in jsonPose]

        ids = [p['id'] for p in jsonPose]

        frame = Drawer.draw_scene(frame, poses, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(33) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 30)


    # When everything done, release the capture
    cap.release()
    cv2.destroyAlWindows()




if __name__ == "__main__":

    print("start frontend")

    port = "7575"

    if len(argv) == 3:
        media = 0 if argv[1] == "webcam" else argv[1]
        start(media, argv[2])
    if len(argv) == 2:
        media = 0 if argv[1] == "webcam" else argv[1]
        start(media, port)
    else:
        start(0, port)

