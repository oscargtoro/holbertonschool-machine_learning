#!/usr/bin/env python3
'''
Displays the number of launches per rocket using the SpaceX API
(https://github.com/r-spacex/SpaceX-API).
'''

import requests as r


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/'
    launches = r.get('{}launches'.format(url)).json()
    rockets = {}
    for launch in launches:
        rocket_name = r.get('{}rockets/{}'.format(
            url, launch['rocket']
        )).json()['name']
        if rocket_name not in rockets:
            rockets.update({rocket_name: 1})
        else:
            rockets.update({rocket_name: rockets[rocket_name] + 1})
    srtd_rockets = sorted(rockets.items(), key=lambda i: i[0])
    srtd_rockets.sort(key=lambda i: i[1], reverse=True)
    for rocket in srtd_rockets:
        print('{}: {}'.format(rocket[0], rocket[1]))
