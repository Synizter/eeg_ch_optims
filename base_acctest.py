from model_set import SugiyamaNet
import capilab_dataset2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import tensorflow as tf
def load(subj):    
    fs = 500
    duration = 2
    sample = fs * duration
    ch = 19
    hp = 0.5
    lp = 40
    data, label = capilab_dataset2.load_target(subj + '_JulyData')
    try:
        x, x_test,  y, y_test = train_test_split(data, label, test_size = .2, stratify = label, random_state = 0)
        x = capilab_dataset2.butterworth_bpf(x, hp, lp, fs)
        x_test = capilab_dataset2.butterworth_bpf(x_test, hp, lp, fs)
        x = np.expand_dims(x, axis = 3)
        x_test = np.expand_dims(x_test, axis = 3)
        # swap sample and channels axis
        # x = np.transpose(x, (0,2,1,3))
        # x_test = np.transpose(x_test, (0,2,1,3))
        
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return None
    else:
        return x, y, x_test, y_test


def step(X, y, x_val, y_val, x_test, y_test, verbose = False):
        tf.random.set_seed(1)
        classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        classifier_loss = tf.keras.losses.CategoricalCrossentropy()
        
        clf = SugiyamaNet(Chans = X.shape[1], Samples = X.shape[2], output_classes = y.shape[1])
        
        clf.compile(optimizer = classifier_optimizer, loss= classifier_loss , metrics=['accuracy'])
        
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=False, save_best_only=True)
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=0)
        clf.fit(X, y,
                batch_size=12, 
                epochs = 15, 
                verbose = verbose, 
                validation_data = (x_val, y_val),
                callbacks = [checkpointer, earlystopper])
        y_preds = clf.predict(x_test, verbose = verbose)
        predicted = np.argmax(y_preds, axis=1)
        ground_truth = np.argmax(y_test, axis=1)
        
        r = accuracy_score(ground_truth, predicted)
        # clf.save('temp_model')
        
        return r
    
def kfold_eval(x,y,x_test, y_test):
    r = np.array([])
    kfold = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 0) #fold
    for i , (train, val) in enumerate(kfold.split(x, np.argmax(y, axis = 1))):
        tf.random.set_seed(1)
        res = step(x[train], y[train], x[val], y[val],x_test, y_test)
        r = np.append(r, res)
    return r, r.mean(), r.std()

def evaluate_base_acc(subj):
    
    #PREPARE DATA ------------------------------------------------------------------------------------------------------------
    maps = {'F4': 0, 'C4': 1, 'P4': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                            'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}
    x, y, x_test,y_test = load(subj) #DATA
    
    print(x.shape)
    d, mean, std = kfold_eval(x, y, x_test, y_test)
    
    
    # with open('mean_sgym.npy', 'wb+') as f:
    #     np.save(f, mean)
    
    # with open('mean_sgym.npy', 'wb+') as f:
    #     np.save(f, std)

    return d, mean, std

if __name__ == "__main__":
    print("Sugiyama", evaluate_base_acc("Sugiyama"))
    print("Lai", evaluate_base_acc("Lai"))
    print("Suguro", evaluate_base_acc("Suguro"))
    print("Takahashi", evaluate_base_acc("Takahashi "))