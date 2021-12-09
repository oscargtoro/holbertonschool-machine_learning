#!/usr/bin/env python3
'''
Module for the method:
    def list_all(mongo_collection)
'''


def list_all(mongo_collection):
    '''
    Lists all documents in a collection.

    Args.
        mongo_collection: pymongo collection object.

    Returns.
        An empty list if no document in the collection.
    '''

    documents = mongo_collection.find()
    return documents
