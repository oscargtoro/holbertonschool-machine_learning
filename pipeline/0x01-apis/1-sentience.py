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

    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'

    while not isinstance(url, type(None)):
        r = requests.get(url).json()
        for species in r['results']:
            sentient = [species['classification'], species['designation']]
            if 'sentient' in sentient:
                if species['homeworld'] is None or \
                        species['homeworld'] in planets:
                    continue
                homeworld = requests.get(species['homeworld']).json()['name']
                planets.append(homeworld)
        url = r['next']

    return planets
