import gzip
import json
import os
import shutil
from os.path import join
from warnings import warn
from contextlib import closing
from functools import wraps
import warnings
import itertools
import inspect
from urllib2 import urlopen, Request

import mipylib.numeric as np

from .base import get_data_home
from ..utils import Bunch

__all__ = ['fetch_openml']

_OPENML_PREFIX = "https://openml.org/"
_SEARCH_NAME = "api/v1/json/data/list/data_name/{}/limit/2"
_DATA_INFO = "api/v1/json/data/{}"
_DATA_FEATURES = "api/v1/json/data/features/{}"
_DATA_QUALITIES = "api/v1/json/data/qualities/{}"
_DATA_FILE = "data/v1/download/{}"


def _get_local_path(openml_path, data_home):
    return os.path.join(data_home, 'openml.org', openml_path + ".gz")

def _retry_with_clean_cache(openml_path, data_home):
    """If the first call to the decorated function fails, the local cached
    file is removed, and the function is called again. If ``data_home`` is
    ``None``, then the function is called once.
    """
    def decorator(f):
        @wraps(f)
        def wrapper():
            if data_home is None:
                return f()
            try:
                return f()
            except HTTPError:
                raise
            except Exception:
                warnings.warn(
                    "Invalid cache, redownloading file",
                    RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f()
        return wrapper
    return decorator


def _open_openml_url(openml_path, data_home):
    """
    Returns a resource from OpenML.org. Caches it to data_home if required.

    Parameters
    ----------
    openml_path : str
        OpenML URL that will be accessed. This will be prefixes with
        _OPENML_PREFIX

    data_home : str
        Directory to which the files will be cached. If None, no caching will
        be applied.

    Returns
    -------
    result : stream
        A stream to the OpenML resource
    """
    def is_gzip(_fsrc):
        return _fsrc.info().get('Content-Encoding', '') == 'gzip'

    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header('Accept-encoding', 'gzip')

    if data_home is None:
        fsrc = urlopen(req)
        if is_gzip(fsrc):
            if PY2:
                fsrc = BytesIO(fsrc.read())
            return gzip.GzipFile(fileobj=fsrc, mode='rb')
        return fsrc

    local_path = _get_local_path(openml_path, data_home)
    if not os.path.exists(local_path):
        try:
            os.makedirs(os.path.dirname(local_path))
        except OSError:
            # potentially, the directory has been created already
            pass

        try:
            with closing(urlopen(req)) as fsrc:
                if is_gzip(fsrc):
                    with open(local_path, 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                else:
                    with gzip.GzipFile(local_path, 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
        except Exception:
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise

    # XXX: First time, decompression will not be necessary (by using fsrc), but
    # it will happen nonetheless
    return gzip.GzipFile(local_path, 'rb')

def _download_data_arff(file_id, sparse, data_home, encode_nominal=True):
    # Accesses an ARFF file on the OpenML server. Documentation:
    # https://www.openml.org/api_data_docs#!/data/get_download_id
    # encode_nominal argument is to ensure unit testing, do not alter in
    # production!
    url = _DATA_FILE.format(file_id)

    @_retry_with_clean_cache(url, data_home)
    def _arff_load():
        with closing(_open_openml_url(url, data_home)) as response:
            if sparse is True:
                return_type = _arff.COO
            else:
                return_type = _arff.DENSE_GEN

            if PY2:
                arff_file = _arff.load(
                    response.read(),
                    encode_nominal=encode_nominal,
                    return_type=return_type,
                )
            else:
                arff_file = _arff.loads(response.read().decode('utf-8'),
                                        encode_nominal=encode_nominal,
                                        return_type=return_type)
        return arff_file

    return _arff_load()