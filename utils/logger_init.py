import logging

file_name = None

def create_dual_loggers(save_file=None):
        
    """This function create loggers

    Returns:
        dual_logger: Logs to both file and console
        file_logger: Logs to file
    """

    # If the target file has changed, then remove all the handlers
    global file_name
    

    if save_file != file_name:
        dual_logger = logging.getLogger('dual_logger')
        if dual_logger is not None and dual_logger.hasHandlers():
            for hdlr in dual_logger.handlers[:]:  # remove all old handlers
                dual_logger.removeHandler(hdlr)

        file_logger = logging.getLogger('file_logger')
        if file_logger is not None and file_logger.hasHandlers():
            for hdlr in file_logger.handlers[:]:  # remove all old handlers
                file_logger.removeHandler(hdlr)


    if logging.getLogger('dual_logger').hasHandlers() and logging.getLogger('file_logger').hasHandlers():
        # Loggers are already configured, no need to configure again
        dual_logger = logging.getLogger('dual_logger')
        file_logger = logging.getLogger('file_logger')

    else:
        # Create a logger for both console and log file
        dual_logger = logging.getLogger('dual_logger')
        dual_logger.setLevel(logging.INFO)

        # Create a console handler and set the level to DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a file handler and set the level to DEBUG
        if save_file is None:
            file_handler = logging.NullHandler()
        else:
            file_handler = logging.FileHandler(save_file)
            file_handler.setLevel(logging.INFO)
        file_name = save_file



        # Create a formatter and add it to the handlers
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Add the handlers to the logger
        dual_logger.addHandler(console_handler)
        dual_logger.addHandler(file_handler)

        # Create a logger for console only
        file_logger = logging.getLogger('file_logger')
        file_logger.setLevel(logging.INFO)

        # Create a console handler and set the level to DEBUG
        if save_file is None:
            file_handler_only = logging.NullHandler()
        else:
            file_handler_only = logging.FileHandler(save_file)
            file_handler_only.setLevel(logging.INFO)


        # Create a formatter and add it to the handler
        file_handler_only.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Add the handler to the logger
        file_logger.addHandler(file_handler_only)

    return dual_logger, file_logger
