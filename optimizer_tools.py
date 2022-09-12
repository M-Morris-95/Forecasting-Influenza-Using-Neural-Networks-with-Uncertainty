import json
import tensorflow as tf
import os

def load_steps(save_dir, optimizer):
    try:
        file = open(save_dir+'optimiser_results.json', 'r')
        res = json.load(file)
        for r in res:
            try: optimizer.register(r['params'], r['target'])
            except Exception as e:
                print(e)
                pass
        print('registered: ', len(optimizer.res))
    except Exception as e:
        print(e)
        print('failed to register previous steps')
        res = []
        pass
    return optimizer

def save_steps(save_dir, optimizer):
    try:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        a_file = open(save_dir+'optimiser_results.json', 'w')
        json.dump(optimizer.res, a_file)
        a_file.close()

        a_file = open(save_dir+'optimiser_max.json', 'w')
        json.dump(optimizer.max, a_file)
        a_file.close()

    except Exception as e:
        print(optimizer.res)
        print(optimizer.max)
        print(e)

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
