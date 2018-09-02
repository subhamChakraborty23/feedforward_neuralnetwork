# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 20:16:25 2018

@author: USER
"""

import pandas as pd
import numpy as np
import math



          
def grad_w(w,b,x,y):
    fx= f(w,b,x)
    return (fx-y)*fx*(1-fx)*w

def grad_b(w,b,x,y):
  fx= f(w,b,x)
  return (fx-y)*fx*(1-fx)
           

              
              

def do_Adam(X,Y,init_w,init_b,max_epoch,eta):
  w,b=init_w,init_b
  w_history,b_history,error_history=[],[],[]
  w,b,mini_batch_size,num_epoch_seen=init_w,init_b,0.1,10
  m_w,m_b,v_w,v_b,eps,beta1,beta2=0,0,0,0,1e-8,0.9,0.99
  
  
  for i in range(max_epoch):
    dw,db=0,0
    for x,y in zip(X,Y):
      dw+=grad_w(w,b,x,y)
      db+=grad_b(w,b,x,y)
    m_w =beta1*m_w+(1-beta1)*dw
    m_b =beta1*m_b+(1-beta1)*db
    v_w = beta2*v_w+(1-beta2)*dw**2
    v_b = beta2*v_b+(1-beta2)*db**2
    m_w=m_w/(1-math.pow(beta1,i+1))
    m_b=m_b/(1-math.pow(beta1,i+1))
    v_w=v_w/(1-math.pow(beta2,i+1))
    v_b=v_b/(1-math.pow(beta2,i+1))
    w=w-(eta/np.sqrt(v_w+eps))*m_w
    b=b-(eta/np.sqrt(v_b+eps))*m_b
    print("Epoch {} : Loss = {} " .format(i, loss(w,b)))
  return w,b  
def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x+b)))   
    
def loss(w,b):
  err=0.0
  for x,y in zip(X,Y):
    fx=f(w,b,x)
    err+=0.5*(fx-y)**2
  return err  

if __name__ =="__main__":
  dataset = pd.read_csv(r'A4_Q7_data.csv')
  X = dataset.iloc[0:,0:1].values
  Y = dataset.iloc[0:,1].values
  max_epoch=100
  init_w=1
  init_b=1
  eta =0.01
  w,b= do_Adam(X,Y,init_w,init_b,max_epoch,eta)
  loss=loss(w,b)
  print("error = {}".format(loss))

