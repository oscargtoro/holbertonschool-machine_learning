#!/usr/bin/env python3
'''
Script that prints the location of a github user
'''

import requests
import sys
from datetime import datetime


if __name__ == "__main__":
    r = requests.get(sys.argv[1])
    if r.status_code == requests.codes.not_found:
        print('Not found')
    if r.status_code == requests.codes.forbidden:
        window_start = int(datetime.now().timestamp())
        window_end = int(r.headers["X-Ratelimit-Reset"])
        print(f'Reset in {(window_end - window_start) // 60} min')
    if r.status_code == requests.codes.ok:
        print(r.json()['location'])
