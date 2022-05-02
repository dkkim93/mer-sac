import yaml
import logging
import git
import torch


def load_config(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters
    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    """Sets python logging
    Args:
        logger_name (str): Specifies logging name
        log_file (str): Specifies path to save logging
        level (int): Logging when above specified level. Default: logging.INFO
    """
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters
    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to get Git repository. Default: "."
    Examples:
        log[args.log_name].info("Hello {}".format("world"))
    Returns:
        log (dict): Dictionary that contains python logging
    """
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./log/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    # Log git information
    repo = git.Repo(path)
    try:
        log[args.log_name].info("Branch: {}".format(repo.active_branch))
    except TypeError:
        pass
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    # Log device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log[args.log_name].info("Device: {}".format(device))

    return log
