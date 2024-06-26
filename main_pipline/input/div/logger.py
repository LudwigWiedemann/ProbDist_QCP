import logging
import main_pipline.input.div.filemanager as file

logger = logging


class Logger(object):
    logger.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    file.create_folder()  # creates Folder to save all data
    filename = "logger"
    logFormatter = logger.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logger.getLogger()
    fileHandler = logger.FileHandler("{0}/{1}.log".format(file.path, filename))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Comment out or remove the console handler to avoid duplicate logs
    # consoleHandler = logger.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)
