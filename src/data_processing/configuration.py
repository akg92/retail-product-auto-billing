
import json
class Configuration:


    def __init__(self,type):
        if not type:
            type="gdrive"
        self.type = type
        with open('conf.json') as f:
            json.load(f)
            self.conf = f[type]




    def get_train_dir(self):
        return self.conf["train_dir"]
    def get_train_old_dir(self):
        return self.conf['train_old_dir']

    def get_data_dir(self):
        return self.conf["data_dir"]