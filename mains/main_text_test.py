# -*- coding: utf-8 -*-
from data_loader.simple_text_data_loader import SimpleTextDataLoader,SimpleCnnTextDataLoader,SimpleTestTextDataLoader
from models.simple_text_model import SimpleLstmTextModel,SimpleCnnTextModel
from trainers.simple_text_trainer import SimpleTextModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from flask import Flask, request

app = Flask(__name__)


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
# data_loader = SimpleCnnTextDataLoader(config)

print('Create the model.')
model = SimpleLstmTextModel(config)
# model = SimpleCnnTextModel(config)

model.load()

testDataLoader = SimpleTestTextDataLoader(config)

(x_train, y_train) = testDataLoader.get_single_feature('你是什么垃圾？')

result = model.model.predict(x_train)

print(result)

result = model.model.predict_classes(x_train)

print(result)



@app.route('/predict/<text>', methods=['GET'])
def index(text):
    '''
    http://0.0.0.0:8000/predict/你算什么垃圾
    :param text:
    :return:
    '''
    # text = request.args.get('text')
    (x_train, y_train) = testDataLoader.get_single_feature(text)

    result = model.model.predict(x_train)

    print(result)

    result = model.model.predict(x_train)

    return 'result {}'.format(result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000)
    # test()
