from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from comet_ml import Experiment

class SimpleTextModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleTextModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            #实现断点续训功能
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                #val_acc->max, val_loss->min, auto
                mode=self.config.callbacks.checkpoint_mode,
                #只保存在验证集上性能最好的模型
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                #只存储权重，不存储结构与配置
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                # 1 打印详细信息，0不打印
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            # tensorboard --logdir=''
            # tensorboard --logdir=experiments/2020-05-09/simple_mnist/logs
            TensorBoard(
                #曲线图输出路径
                log_dir=self.config.callbacks.tensorboard_log_dir,
                #是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True，日志文件会变得非常大
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        # if hasattr(self.config,"comet_api_key"):
        #     experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #     experiment.disable_mp()
        #     experiment.log_multiple_params(self.config)
        #     self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
