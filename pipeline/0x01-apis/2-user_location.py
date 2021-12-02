#!/usr/bin/env python3
'''
Script that prints the location of a github user
'''

import requests
import sys
import time


if __name__ == "__main__":
    params = {'Accept': "application/vnd.github.v3+json"}
    r = requests.get(sys.argv[1], params=params)
    if r.status_code == requests.codes.not_found:
        print('Not found')
    if r.status_code == requests.codes.forbidden:
        window_start = int(time.time())
        window_end = int(r.headers['X-Ratelimit-Reset'])
        print('Reset in {} min'.format((window_end - window_start) // 60))
    if r.status_code == requests.codes.ok:
        print(r.json()['location'])
