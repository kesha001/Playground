import os, requests

def validate(request):
    if "Authorization" not in request.headers:
        return None, ("missing credentials", 401)

    token = request.headers["Authorization"]

    if not token:
        return None, ("missing credentials", 401)

    # passing authorization token along to validate request
    response = requests.post(
        f"http://{os.environ.get('AUTH_SVC_ADDRESS')}/validate",
        headers={"Authorization": token},
    )

    # response.txt will contain body which will be access that the bearer of this token has
    if response.status_code == 200:
        return response.txt, None
    else:
        return None, (response.txt, response.status_code)