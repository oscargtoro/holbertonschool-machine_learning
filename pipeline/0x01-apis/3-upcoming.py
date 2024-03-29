#!/usr/bin/env python3
'''
Script that displays the upcoming launch using the SpaceX API
(https://github.com/r-spacex/SpaceX-API)
'''

import requests
import time


if __name__ == "__main__":
    now = int(time.time())
    upcoming = {}
    dates = []
    url = 'https://api.spacexdata.com/v4/'
    launches = requests.get('{}launches/upcoming'.format(url)).json()
    for launch in launches:
        # if launch['date_unix'] < now:
        #     continue
        if not upcoming:
            upcoming = launch
        else:
            if launch['date_unix'] == upcoming['date_unix']:
                continue
            if launch['date_unix'] < upcoming['date_unix']:
                upcoming = launch
    rocket = requests.get('{}rockets/{}'.format(
        url, upcoming["rocket"])
    ).json()

    launchpad = requests.get(
        '{}launchpads/{}'.format(url, upcoming["launchpad"])
    ).json()

    info = '{} ({}) {} - {} ({})'.format(
        upcoming['name'],
        upcoming['date_local'],
        rocket['name'],
        launchpad['name'],
        launchpad['locality']
    )
    print(info)
