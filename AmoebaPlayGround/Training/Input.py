import argparse
from datetime import datetime


def get_model_filename():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file-name', action="store",
                        dest="log_file_name", default="")
    args = parser.parse_args()
    log_file_name = args.log_file_name
    if log_file_name is None or log_file_name == "":
        date_time = datetime.now()
        log_file_name = date_time.strftime("%Y-%m-%d_%H-%M-%S")

    return log_file_name
