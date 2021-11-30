#!/usr/bin/env python3
'''
Module for the function(s):
    def availableShips(passengerCount)
'''

import requests
import json
import locale


def availableShips(passengerCount):
    '''
    Accesses the Swapi API for ships that can hold a given number of passengers

    Args.
        passemgerCount: Number of passengers.

    Returns.
        A list of ships that can hold a given number of passengers.
    '''

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    url = 'https://swapi-api.hbtn.io/api/starships/'
    r = requests.get(url)

    ships = r.json()
    avail_ships = []

    while ships['next'] is not None:
        for ship in ships['results']:
            try:
                ship['passengers'] = locale.atoi(ship['passengers'])
            except ValueError:
                continue
            if ship['passengers'] >= passengerCount:
                avail_ships.append(ship['name'])
        ships = requests.get(ships['next']).json()

    return avail_ships
