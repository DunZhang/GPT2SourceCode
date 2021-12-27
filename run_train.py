# +++++++++++++++
# ---------------
import logging
from Train import train
from utils import create_logger
from GPT2ChatbotConf import GPT2ChatbotConf

if __name__ == "__main__":
    conf = GPT2ChatbotConf()
    logger = create_logger(conf)

    train(conf)
