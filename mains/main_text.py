# -*- coding: utf-8 -*-
from data_loader.simple_text_data_loader import SimpleTextDataLoader
from models.simple_text_model import SimpleTextModel
from trainers.simple_text_trainer import SimpleTextModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)
        config = process_config("../configs/simple_text_config.json")
    except Exception as e:
        print('str(e):\t\t', str(e))
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.model_dir])

    print('Create the data generator.')
    data_loader = SimpleTextDataLoader(config)

    print('Create the model.')
    model = SimpleTextModel(config)

    print('Create the trainer')
    trainer = SimpleTextModelTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
