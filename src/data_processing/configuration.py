
import json
import os
class Configuration:


    def __init__(self,type=None):
        dir_path =self.get_current_path()
        conf_path = os.path.join(dir_path,'conf.json')
        if not type:
            type="gdrive"
        self.type = type
        with open(conf_path,'r') as f:
            json.load(f)
            self.conf = f[type]



    def get_current_path(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return dir_path
    def get_train_dir(self):
        return self.conf["train_dir"]
    def get_train_old_dir(self):
        return self.conf['train_old_dir']

    def get_data_dir(self):
        return self.conf["data_dir"]