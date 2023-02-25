import jwt, datetime, os
from flask import Flask, request
from flask_mysqldb import MySQL

# The flask object implements a WSGI application and acts as the central object. 
# Acts as a central registry for the view functions, the URL rules, template configuration and much more
server = Flask(__name__)
# Makes that our app can connect to mysql db
mysql = MySQL(server)

# config
# variables to connect to mysql database
# get mysql host from our environment; at beggining os.environ.get("MYSQL_HOST") is none; export MYSQL_HOST=localhost (in terminal)  -> localhost
server.config["MYSQL_HOST"] = os.environ.get("MYSQL_HOST")
server.config["MYSQL_USER"] = os.environ.get("MYSQL_USER")
server.config["MYSQL_PASSWORD"] = os.environ.get("MYSQL_PASSWORD")
server.config["MYSQL_DB"] = os.environ.get("MYSQL_DB")
server.config["MYSQL_PORT"] = os.environ.get("MYSQL_PORT")


@server.route("/login", methods=["POST"])
def login():
    """
    Checks database for user data that is trying to log in
    returns 401 missing credentials if provided credentials for user are not contained in db
    """

    # authorization attribute provides credentials from basic authentification header (auth.username and auth.password)
    auth = request.authorization
    if not auth:
        return "missing credentials", 401
    
    # check db for username and password
    cur = mysql.connection.cursor()
    # using created cursor to execute queries
    # %s - whatever email is passed into request (username = email)
    res = cur.execute(
        "SELECT email, password FROM user\
        WHERE email=%s", (auth.username,)
    ) # res - array of rows 
    
    # at least 1 row with given username 
    if res > 0:
        # set the row that contains user data to cursor
        user_row = cur.fetchone()
        email = user_row[0]
        password = user_row[1]

        if auth.username != email or auth.password != password:
            return "invalid credentials", 401
        else:
            return createJWT(auth.username, os.environ.get("JWT_SECRET"), True)

    else:
        return "invalid credentials", 401


@server.route("/validate", methods=["POST"])
def validate():
    """
    Gets jwt token in Authorization header, takes <credentials> part from jwt and decodes with secret key
    returns "not authorized", 403 in case of wrong key, otherwise returns decoded jwt and 200 code`
    """

    # jwt token should be in Authorization header
    encoded_jwt = request.headers['Authorization']

    # if jwt is not present in header return error
    if not encoded_jwt:
        return "missing credentials", 401

    # We require our jwt be in format <type> <credentials>, specifically Bearer <credentials>
    encoded_jwt = encoded_jwt.split(" ")[1]

    # decoding with secret key
    try:
        decoded = jwt.decode(
            encoded_jwt, os.environ.get("JWT_SECRET"), algorithms=["HS256"]
        )
    except:
        return "not authorized", 403

    return decoded, 200


def createJWT(username, secret, authz: bool):
    """
    Fuction accepts claims of the user, private key, and if the user is admin
    Returns Jwt token based on the info
    """
    return jwt.encode(
            payload={
                "username": username,
                "exp": datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1), # set expiration of token 1 day
                "iat": datetime.datetime.utcnow(), # when the token is issued
                "admin": authz,
            },
            key=secret,
            algorithm="HS256",
            )


if __name__ == "__main__":
    # host 0.0.0.0 allows to listen to any address available on our host (to all public IPs), the default is localhost
    # the host is our docker container and it can have different if addresses within docker network so we use 0.0.0.0 
    # using local host, requests sent from outside of conitainer will never make to our flask app
    server.run(host="0.0.0.0", port=5000)