
import configparser
#import jso

import os

from datetime import datetime
#from readConfig import read_config
import hashlib


def detect_images(PATH):
    model = DeepFace.build_model("Facenet")
    print(PATH)
    instances = [] 

    #print(instance)
    
    return instance

def read_config(database_name):
    config = configparser.ConfigParser()
    #config.read('../config.ini')
    config.read('config.ini')
    return config[database_name]


#Sample Usage
if __name__ == "__main__":
    # Specify the path to the image you want to process
    image_path = "../test_dataset/img58.jpg"
    
    # Detect information from the image
    instance = detect_images(image_path)
    
    # Connect to the database and insert the detected information
    #connect_database('database', instance)
