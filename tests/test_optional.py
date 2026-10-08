import numpy as np, pandas as pd,pytest
from test_notebooks import definitions

def test_linear_svm_separates_synthetic_points():
    ns=definitions('hw3/practical1_svm.ipynb',{'np':np,'pd':pd})
    X=pd.DataFrame([[-2,0],[-1,0],[1,0],[2,0]],columns=['x','y'])
    pred=ns['LinearSVM']({'x':1,'y':0},0)(X)
    assert pred.tolist()==[-1,-1,1,1]
    assert ns['validate'](pd.Series([-1,-1,1,1]),pred,verbose=False)==1

def test_tensorflow_models_train_on_synthetic_batch():
    tf=pytest.importorskip('tensorflow')
    ns={'np':np,'tf':tf,'models':tf.keras.models,'layers':tf.keras.layers,
        'regularizers':tf.keras.regularizers,'INPUT_SHAPE':(784,),'NUM_CLASSES':10,
        'num_classes':10,'Sequential':tf.keras.Sequential,'Flatten':tf.keras.layers.Flatten,
        'Dense':tf.keras.layers.Dense,'Callback':tf.keras.callbacks.Callback}
    ns.update({name:getattr(tf.keras.optimizers,name) for name in ('Adagrad','RMSprop','Adam','SGD')})
    definitions('hw3/practical2.ipynb',ns)
    rng=np.random.default_rng(42);flat=rng.random((4,784),dtype=np.float32)
    y=tf.keras.utils.to_categorical([0,1,2,3],10)
    for name in ('create_baseline_model','create_dropout_model','create_l2_model','create_batch_norm_model'):
        model=ns[name]();model.compile(optimizer='adam',loss='categorical_crossentropy')
        assert np.isfinite(model.train_on_batch(flat,y))
        assert model(flat,training=False).shape==(4,10)
    model=ns['model_creator'](optimizer='adam',num_units=8)
    assert np.isfinite(model.train_on_batch(flat.reshape(4,28,28),y)).all()
