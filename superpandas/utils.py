try:
    # python2
    from urlparse import urlparse
except ModuleNotFoundError:
    # python3
    from urllib.parse import urlparse

def valid_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False