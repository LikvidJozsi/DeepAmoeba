import os

from typing.io import IO

from AmoebaPlayGround.Evaluator import fix_reference_agents


logs_folder = 'Logs/'
field_separator = ','

class Logger:
    def new_episode(self):
        pass

    def log(self, key, message):
        pass


class FileLogger(Logger):
    def __init__(self, log_file_name):
        if log_file_name is None or log_file_name == "":
            raise Exception("Bad string received.")

        self.log_file_name = log_file_name
        self.log_file_path = self.log_file_name + ".csv"
        self.log_file_path = os.path.join(logs_folder, self.log_file_path)
        self.headers = []
        self.field_names = []
        self.field_values = []

    def log(self,key, value):
        self.field_names.append(key)
        self.field_values.append(str(value))


    def new_episode(self):
        with open(self.log_file_path, mode="a", newline='') as log_file:
            if len(self.headers) == 0:
                self.headers = self.field_names
                self._write_line(log_file,self.headers)
            self._write_line(log_file,self.field_values)
            self.field_names = []
            self.field_values = []

    def validate_fields(self,headers,field_names):
        for field_name in field_names:
            if field_name not in headers:
                raise Exception("Field name not found among headers: " + field_name)

    def _write_line(self,file: IO, fields):
        fields_string = field_separator.join(fields)
        file.write(fields_string + "\n")



class ConsoleLogger(Logger):
    def log(self,key, message):
        print(key + ": " + str(message))