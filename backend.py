from http.server import BaseHTTPRequestHandler,HTTPServer
from sys import argv
import numpy as np
import cv2
import time

from src.system.interface import AnnotatorInterface


"""

Start the backend.py with the port and the maximum number of people (default : port 7575, 1 person).
Send a post request that contains the image to annotate (as made in front_end_2d).
Both get and post request return the following JSON format :


List<annotation> with annotation having the following attributes
id : person id
pose_2d : dictionary of joint name => dictionary {x,y} (same joint names as src.utils.PoseConfig.NAMES)
pose_3d : same structure as pose2d with the third dimension
confidence : a value indicating the confidence for the overall pose 2d


To run on an external server :
1.- run the backend on the server
2.- redirect the backend selected port to your local computer using :
ssh -i ~/.ssh/server_private_key -f user@ip -L port_frontend:localhost:port_backend -N
3.- start the frontend on localhost:port_frontend
"""



class PoseApiHttpHandler(BaseHTTPRequestHandler):

    annotator = None



    def do_GET(self):

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(PoseApiHttpHandler.annotator.jsonify().encode("utf8"))



    def do_POST(self):


        length = int(self.headers['content-length'])

        img = self.rfile.read(length)

        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        PoseApiHttpHandler.annotator.update(img)

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self.wfile.write(PoseApiHttpHandler.annotator.jsonify().encode("utf8"))




def run(port=7575, max_persons=2):

    PoseApiHttpHandler.annotator = AnnotatorInterface.build(max_persons)

    server_address = ('', port)

    httpd = HTTPServer(server_address, PoseApiHttpHandler)
    print('Ready to serve requests on localhost:' + str(port))

    httpd.serve_forever()



if __name__ == "__main__":
    print("load system")
    if len(argv) == 3:
        run(port=int(argv[1]), max_persons=int(argv[2]))
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()