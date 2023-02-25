import pika, json

def upload(f, fs, channel, access):
    """
    First upload the file to mongodb using gridfs, after uploading put a message in rabbitmq queue
    so that a downstream service when they pull that message from the queue can process the upload
    by pulling it from mongodb
    """

    # put a file file into mongodb
    try:
        # file id
        fid = fs.put(f)
    except Exception as err:
        return "internal server error", 500
    
    message = {
        "video_fid": fid,
        "mp3_fid": None, # later it is assigned
        "username": access["username"],
    }

    # put the message into queue
    try:
        channel.basic_publish(
            exchange=""
        )