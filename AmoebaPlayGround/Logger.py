import os

from AmoebaPlayGround.Evaluator import fix_reference_agents


class Logger:
    def log(self, message):
        pass

    def newline(self):
        self.log("\n")

    def log_newline(self, message):
        self.log(message)
        self.newline(self)

    def log_value(self, message):
        pass

    def close(self):
        pass


class FileLogger(Logger):
    def __init__(self, log_file_name):
        if log_file_name is None or log_file_name == "":
            raise Exception("Bad string received.")

        self.log_file_name = log_file_name
        self.logs_folder = 'Logs/'
        self.log_file_path = self.log_file_name + ".log"
        self.log_file_path = os.path.join(self.logs_folder, self.log_file_path)
        self.log_file = open(self.log_file_path, mode="a+", newline='')

    def log(self, message):
        self.log_file.write(message)
        # So that interrupt won't delete full content.
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class AmoebaTrainingFileLogger(FileLogger):
    def __init__(self, log_file_path):
        super().__init__(log_file_path)

        self.log("episode\taverage_game_length\tloss\trating\t")
        for reference_agent in fix_reference_agents:
            self.log("%s\t" % reference_agent.name)

        self.newline()

    def log_value(self, message):
        self.log(f"{message}\t")


# class ConsoleLogger(Logger):
#     def log(self, message):
#         print(message)
#
#
# class AmoebaTrainingConsoleLogger(ConsoleLogger):
#     def log_value(self, message):
#         print("WRITE SOMETHING")