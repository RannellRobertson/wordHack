import requests


class BadRequestError(Exception):
    pass


class InternalServerError(Exception):
    pass


class ForbiddenRequest(Exception):
    pass


def fetch(filename, url, **kwargs):
    with open(filename, "wb") as f:
        response = requests.get(url, auth=kwargs.get("auth", None), params=kwargs.get("params", None))
        try:
            if response.status_code == 200:
                f.write(response.text.encode("utf-8"))
        except response.status_code == 400:
            raise BadRequestError("Bad Request Error was caught. "
                                  "Check to be sure the url is correct")
        except response.status_code == 403:
            raise ForbiddenRequest("Forbidden Request Error was caught. "
                                   "Unauthorized request was made")
        except response.status_code == 404:
            raise FileNotFoundError("File Not Found Error was caught. "
                                    "File cannot be found")
        except response.status_code == 500:
            raise InternalServerError("Internal Server Error was caught. "
                                      "Server cannot handle")


def open_file(path):
    with open(path) as f:
        text = f.read()
        return text.lower()
