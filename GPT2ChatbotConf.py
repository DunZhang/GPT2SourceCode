"""
配置类
"""


class GPT2ChatbotConf():
    def __init__(self):
        # root dir
        self.data_model_dir = "G:/PROJ_DATA_MODEL/nlpgeneration"
        self.output_dir = "G:/PROJ_OUTPUT"
        # logger
        self.logger_name = "gpt2chat"
        self.log_path = "log.txt"
        # train data
        self.train_data_path = "LCCC-base_train.json"
        self.max_len = 200
        # model
        self.pretrained_model_dir = "Novel_GPT_ZD"
        self.lr = 5e-5

        # save
        self.save_model_dir = "saved_models"

        # train
        self.log_step = 100
        self.save_step = 50000
        self.batch_size = 8
        self.use_multi_proc = False
        self.device = "0"
        self.epoch = 20
