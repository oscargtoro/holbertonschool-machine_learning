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
    url = 'https://api.spacexdata.com/v4/'
    launches = requests.get(f'{url}launches/upcoming').json()
    for launch in launches:
        if launch['date_unix'] < now:
            continue
        print(launch['date_local'])
        if not upcoming:
            upcoming = launch
        else:
            if launch['date_unix'] - now < upcoming['date_unix'] - now:
                upcoming = launch
    launch_name = upcoming['name']
    launch_date = upcoming['date_local']
    rocket = requests.get(f'{url + "rockets/" + upcoming["rocket"]}').json()
    launchpad = requests.get(
        f'{url + "launchpads/" + upcoming["launchpad"]}'
    ).json()
    info = '{} ({}) {} - {} ({})'.format(
        upcoming['name'],
        upcoming['date_local'],
        rocket['name'],
        launchpad['name'],
        launchpad['locality']
    )
    print(info)
