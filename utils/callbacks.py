import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class AdvancedEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

      Arguments:
          patience: Number of epochs to wait after min has been hit. After this
            number of no improvement, training stops.
          filter_order: The order of the filter to filter the loss
          decay_rate: When the loss stops decreasing, set the learning rate 'lr'
            to 'lr'/'decay_rate'. If 'decay_rate' is None, stop training when the
            loss stops decreasing
          min_lr: If learning rate is less than 'min_lr', stop training. Never stop
            training is 'min_lr' is None
          log_dir: Path to write the filtered loss and learning rate for tensorboard

    """
    def __init__(self, patience=3, filter_order=3, decay_rate=None, min_lr=None, log_dir=None):
        super(AdvancedEarlyStopping, self).__init__()

        self.patience = patience
        self.filter_order = filter_order
        self.decay_rate = decay_rate
        self.stack = []
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.lr = np.nan
        self.best_weights = None
        if min_lr is None:
            self.min_lr = -np.inf
        else:
            self.min_lr = min_lr
        self.file_writer = None
        if log_dir is not None:
            self.file_writer = tf.summary.create_file_writer(log_dir + '/metrics')

    def on_train_begin(self, logs=None):
        # Init param
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.best_weights = self.model.get_weights()
        if self.decay_rate is not None:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            else:
                self.lr = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        # Update stack
        if len(self.stack) < self.filter_order:
            self.stack.append(current)
        else:
            self.stack.pop(0)
            self.stack.append(current)
        # Filter 'loss'
        mean = sum(self.stack)/len(self.stack)
        # Loss is decreasing
        if mean < self.best:
            self.best = mean
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            # Loss stops decreasing
            self.wait += 1
            if self.wait >= self.patience:
                # Reach the patience
                if self.decay_rate is not None:
                    # If auto learning rate decay is set
                    self.lr = float(K.get_value(self.model.optimizer.lr))
                    # Decay and update the learning rates
                    self.lr /= self.decay_rate
                    K.set_value(self.model.optimizer.lr, self.lr)
                    if self.lr < self.min_lr:
                        # Reach the 'min_lr' limit
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                    # reset wait time
                    self.wait = 0
                    # reset best loss
                    self.best = np.inf
                    # Restoring model weights from the end of the best epoch
                    self.model.set_weights(self.best_weights)
                else:
                    # No learning rate schedule. Stop training
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    # Restoring model weights from the end of the best epoch
                    self.model.set_weights(self.best_weights)
        if self.decay_rate is not None:
            print('Best loss (filtered): %.5f.' % self.best,
                  'Current loss (filtered): %.5f.' % mean, 'Current learning rate: ' + str(self.lr) + '.')
        else:
            print('Best loss (filtered): %.5f.' % self.best, 'Current loss (filtered): %.5f.' % mean)

        # Write log file
        if self.file_writer is not None:
            with self.file_writer.as_default():
                tf.summary.scalar('filtered loss', data=mean, step=epoch)
                if self.decay_rate is not None:
                    tf.summary.scalar('learning rate', data=self.lr, step=epoch)


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.decay_rate is not None:
                if self.lr < self.min_lr:
                    print('Epoch %05d: early stopping.' % (self.stopped_epoch + 1), 'Learning rate reaches limit.')
                else:
                    print('Epoch %05d: early stopping.' % (self.stopped_epoch + 1), 'Wait time reaches limit.')
            else:
                print('Epoch %05d: early stopping.' % (self.stopped_epoch + 1), 'Wait time reaches limit.')



class LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):

    def __init__(self, schedule):
        """
        Auto decay learning rate.
        Arguments:
            schedule: a list of integers. In which epoch the learning rate should be decayed.
        """

        self._schedule = schedule

        def lr_decay(epoch, lr):
            if epoch in schedule:
                lr *= 0.1
            return lr

        super().__init__(lr_decay, 1)

