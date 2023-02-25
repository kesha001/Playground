import os, requests

def login(request):
    auth = request.authorization
    if not auth:
        return None, ("missing credentials", 401)

    # basic auth header
    basicAuth = (auth.username, auth.password)

    response = request.post(
        f"http://{os.environ.get('AUTH_SVC_ADDRESS')}/login",
        auth=basicAuth,
    )

    # response.txt is our token
    if response.status_code == 200:
        return response.txt, None
    else:
        return None, (response.txt, response.status_code)

