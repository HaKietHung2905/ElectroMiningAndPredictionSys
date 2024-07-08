import configparser

def read_config(database_name):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[database_name]

