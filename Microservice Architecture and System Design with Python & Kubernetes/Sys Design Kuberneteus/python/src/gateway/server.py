import os, gridfs, pika, json
from flask import Flask, request
# mongodb to store out files
from flask_pymongo import PyMongo
from auth import validate
from auth_svc import access
from storage import util

server = Flask(__name__)
# 27017 - default mongodb port, database will be called videos, 
# host.minicube.internal lets access to localhost from withing our kuberneteus cluster
# this MONGO_URI will be endpoint to interface with our mongodb
server.comfig["MONGO_URI"] = "mongodg://host.minicube.internal:27017//videos"

# manages mongodb connections for flask app
mongo = PyMongo(server)

# wrap mongodb videos data base in gridfs
# gridfs allows to work with files larges 16 mb (mongodb limitation)
# divides data into chunks no larger than 16 mb each
# GridFS stores files in two collections: "chunks" stores the binary chunks and "files" stores the file's metadata
fs = gridfs.GridFS(mongo.db)

# setup rabbitmq connection with rabbitmq name
# this queue allows asynchronisity that allows not to wait for an internal service to process the video
# before being able to return a responce to the client
connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

@server.route("/login", methods=["POST"])
def login():
    token, err = access.login(request)

    if not err:
        return token
    else:
        return err

@server.route("/upload", methods=["POST"])
def upload():
    access, err = validate.token(request)

    # deserializes json into python object (our claims from token)
    access = json.loads(access)

    if access["admin"]:
        # request.files will have a key for the file which will be defined when we send the request and the actual file as value
        if len(request.files) > 1 or len(request.files) < 0:
            return "Exactly 1 file required", 400

        for _, f in request.files:
            # passing file, gridfs, rabbitmq channel and access, in case error returns it otherwise none
            err = util.upload(f, fs, channel, access)

            if err:
                return err

        return "success!", 200
    else: 
        return "not authorized", 401

@server.route("/download", methods=["GET"])
def download():
    pass


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8080)