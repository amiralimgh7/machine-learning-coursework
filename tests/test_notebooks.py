"""Exercise numerical functions on synthetic data, without training full homework sweeps."""
from pathlib import Path
import ast,json,numpy as np,pandas as pd,torch
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
def definitions(path,namespace):
    notebook=json.loads((ROOT/path).read_text(encoding='utf-8'))
    for c in notebook['cells']:
        if c['cell_type']!='code':continue
        source=''.join(c['source'])
        source='\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('%','!')))
        tree=ast.parse(source)
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        for node in selected:exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace
def test_logistic_gradient_matches_finite_differences():
    ns=definitions('hw2/MLE_MAP_LR.ipynb',{'np':np,'pd':pd})
    X=np.array([[1.,2.],[-1.,1.],[3.,-2.]])
    y=np.array([1.,0.,1.]);w=np.array([.1,.2]);b=.3
    gradient=ns['gradient_cost_function'](X,y,w,b,lambda_=0.1)
    assert len(gradient)==2
    # Notebook returns (dw, db), according to the original implementation.
    dw,db=gradient
    eps=1e-6
    for i in range(2):
        plus=w.copy();minus=w.copy();plus[i]+=eps;minus[i]-=eps
        finite=(ns['J_wb'](X,y,plus,b,.1)-ns['J_wb'](X,y,minus,b,.1))/(2*eps)
        assert np.isclose(dw[i],finite,atol=1e-5)
    finite=(ns['J_wb'](X,y,w,b+eps,.1)-ns['J_wb'](X,y,w,b-eps,.1))/(2*eps)
    assert np.isclose(db,finite,atol=1e-5)
    assert np.isfinite(ns['sigmoid'](np.array([-1000.,1000.]))).all()
def test_pinn_loss_has_finite_parameter_gradients():
    ns=definitions('hw2/pinn.ipynb',{'np':np,'torch':torch,'nn':torch.nn,'N':3,'alpha':1.0,'x0':.1})
    model=ns['PINN'](3,num_hidden_layers=2,num_neurons=8)
    t=torch.linspace(0,1,6).reshape(-1,1)
    loss=ns['physics_informed_loss'](model,t)+ns['initial_condition_loss'](model,t)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
def test_custom_color_conversion_and_constant_histogram():
    ns=definitions('computer-vision/color_processing.ipynb',{'np':np})
    colors=np.array([[[255,0,0],[0,255,0],[0,0,255]]],dtype=np.uint8)
    hsv=ns['RGB_to_HSV'](colors)
    assert np.allclose(hsv[0,:,0],[0,1/3,2/3])
    assert np.allclose(hsv[0,:,1:],1)
    equalized=ns['histogram_equalization'](np.full((4,4),50,dtype=np.uint8))
    assert equalized.shape==(4,4) and np.isfinite(equalized).all()
