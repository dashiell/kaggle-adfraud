import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import gc
#from model import models
from sklearn.model_selection import KFold, StratifiedKFold
from keras import backend as K
import time



def create_keras_meta(model, batch_size, n_folds, X_train, X_test, y_train):
    '''
    Use like this:

    train_meta, test_meta = train_utils.create_keras_meta(model, 64, 20, X_train, X_test, y_train)
    
    where model is the compiled keras model
    
    '''
    mean_auroc = 0
    n_classes = y_train.shape[1]
    
    train_meta = np.zeros(y_train.shape)
    test_meta = np.zeros((X_test.shape[0], n_classes))
    initial_weights = model.get_weights()
    
    
    if(n_classes > 1):
        kf = KFold(n_splits = n_folds, shuffle=True, random_state=888)
        splits = kf.split(X_train)
    else:
        kf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=888)
        splits = kf.split(X_train, y_train)


    for fold, (train_ix, valid_ix) in enumerate(splits):
        startTime = time.time()
        
        x1,x2 = X_train[train_ix], X_train[valid_ix]
        y1,y2 = y_train[train_ix], y_train[valid_ix]
        
        val_auroc, epochs = fit_on_val(model, batch_size, x1,x2,y1,y2)
        
        mean_auroc += 1/n_folds * val_auroc
        
        oof_preds = model.predict(x2)
        train_meta[valid_ix] = oof_preds
        
        test_preds = model.predict(X_test)    
        test_meta += 1/n_folds * test_preds
    
        del x1,x2,y1,y2; gc.collect()
        
        model.set_weights(initial_weights)
        print('finished fold', fold, 'auroc', val_auroc, 'time', time.time() - startTime, 'epochs', epochs)
    
    # clear the TF graph, release memory
    K.clear_session()
    
    return train_meta, test_meta, mean_auroc


def fit_on_val(model, batch_size, X_train, X_valid, y_train, y_valid):
    """train until validation loss stops decreasing   
    """
           
    best_weights = None
    val_auroc = -np.inf
    best_epochs = -np.inf
    ttl_epochs = 0
    
    epochs_since_improve = 0
    
    while epochs_since_improve < 3:
        model.fit(X_train, y_train, batch_size, verbose = 1)
        y_preds = model.predict(X_valid, batch_size)
        
        
        current_auroc = roc_auc_score(y_valid, y_preds)
        ttl_epochs +=1
        # NOTE: i'm maximizing AUC, so you probably want to change the sign
        # if you're tring to minimize RMSE or something..
        if current_auroc > np.round(val_auroc, 4):
            
            val_auroc = current_auroc
            best_epochs = ttl_epochs
            best_weights = model.get_weights()
            print('~~val auroc ', current_auroc, best_epochs)
            epochs_since_improve = 0

        else:
            epochs_since_improve += 1
            
    
    model.set_weights(best_weights)

    print("returning...", val_auroc, "epcohs", best_epochs)    
    
    return val_auroc, best_epochs
    