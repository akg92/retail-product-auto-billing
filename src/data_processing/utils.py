import os

"""
    create the folder and capture the error.
"""
def create_folder_if_not_exist(folder):

    try:
        os.mkdir(folder)
    except:
        pass
