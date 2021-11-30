#!/usr/bin/env python3
'''
Module for the function(s):
    def sentientPlanets()
'''

import requests


def sentientPlanets():
    '''
    Queries the Swapi API for names of the home planets of all sentient species

    Returns:
        A list with the names of the home planets of all sentient species.
    '''

    url = 'https://swapi-api.hbtn.io/api/species/'
    r = requests.get(url).json()
    planets = []

    while r['next'] is not None:
        for species in r['results']:
            if species['designation'] == 'sentient':
                if species['homeworld'] is None or species['homeworld'] in planets:
                    continue
                homeworld = requests.get(species['homeworld']).json()['name']
                planets.append(homeworld)
        r = requests.get(r['next']).json()

    return planets
