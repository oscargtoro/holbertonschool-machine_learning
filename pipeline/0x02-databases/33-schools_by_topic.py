#!/usr/bin/env python3
'''
Module for the method
    def schools_by_topic(mongo_collection, topic):
'''


def schools_by_topic(mongo_collection, topic):
    '''
    Searches schools having a specific topic.

    Args.
        mongo_collection: pymongo collection object.
        topic: (string) topic to search.

    Returns.
        A list of schools having a specific topic.
    '''

    return mongo_collection.find( {'topics': {'$in': [topic]} } )
