#! /Volumes/Environment/HeadRepo/.venv/bin/python
# coding: utf-8

# # Схемы дифференцирования
# 
# Схемы дифференцирования используются для решения дифференциальных уравнений и численного вычисления производных. 

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Вид схемы дифференцирования
# 
# Общий вид схемы дифференцирования может быть записан, как
# $$
# f^{(k)}(x) \approx a_1 f(x_1) + a_2 f(x_2) +..+ a_n f(x_n)
# $$
# где коэффициенты $a_1,..,a_n$ вычисляются в зависимости от того, как расположены друг относительно друга точки $x,x_1,..,x_n$. Подробно мы это разбирали на семинаре и на лекциях.
# 
# Предположим, мы построили на бумаге некую схему дифференцирования, остаётся её удобно реализовать в программе.

# In[ ]:


class FinDifferenceDerivative:
    def __init__(self, coefs, xs, order=1):
        '''
        coefs  [n] float32 -- массив коэффициентов схемы
        xs [n] float32 -- массив точек x_k, в которых даются значения функции (относительно x)
        order int -- порядок производной, которую схема приближённо вычисляет
        '''
        self.coefs = coefs
        self.xs = xs
        self.order = order

    def giveInfo(self):
        '''
        Печатает общую информацию о схеме
        '''
        print("Схема вычисляет производную порядка "+str(self.order))
        print("Коэффициенты: "+str(self.coefs))
        print("Точки(относительно x): "+str(self.xs))

    def compute(self, fs):
        '''
        Вычисляет значение производной в точке  x, используя значения функции
        fs [...,n] float32 -- тензор значений функции, последняя размерность
            должна быть равна числу требуемых значений
        РЕЗУЛЬТАТ: тензор [...,1] вычисленных производных
        '''
        pass


# In[ ]:


#ТЕСТЫ
def ftest(x):
    return np.exp(x**2)

h=0.0001
rightDiff = FinDifferenceDerivative(coefs=np.array([-1,1])/h,xs=np.array([0,1])*h)

inp=ftest(np.array([1,1+h]))
assert np.abs(rightDiff.compute(inp) - 2*np.exp(1) )<1e-3 

tensorInp = np.concatenate([ftest(np.array([1,1+h])[np.newaxis,:]),\
                         ftest(np.array([1,1+h])[np.newaxis,:]) ], axis=0)
assert np.all(np.abs(rightDiff.compute(tensorInp) - 2*np.exp(1)*np.ones([2]))<1e-3 )
rightDiff.giveInfo()


# ## Проверка качества аппроксимации
# 
# 

# In[ ]:


hs = np.arange(3e-5,0.7,5e-3)
x=0.1


def ftest(x):
    return np.exp(-x**2)
def ftestPrime(x):
    return -2*x*np.exp(-x**2)

#YOUR CODE, write a routine to compute derivative for different values of h
diffOutputsRight = np.zeros([len(hs)])
for k in np.arange(len(hs)):
    #give an argument list
    xs = #?
    differentiator = #create Differentiator
    diffOutputsRight[k] = #compute the derivative


# In[ ]:


f,ax = plt.subplots(figsize=(8,5))

ax.set_title("Погрешности вычисления, формула правых разностей")
ax.set_xlabel("h")
ax.set_ylabel("Err")
ax.grid()
ax.plot(hs,np.abs(diffOutputsRight-ftestPrime(x)))


# Давайте используем более точную формулу центральных разностей.

# In[ ]:


diffOutputsCent = np.zeros([len(hs)])

#YOUR CODE
for k in np.arange(len(hs)):
    xs = np.array([x-hs[k],x+hs[k]])#!!!!   f'(x) ~~ ( f(x+h)-f(x-h) ) /2h  Central Difference

    differentiator = #??? create a central-difference Differentiator

    diffOutputsCent[k] = #


# 
# 

# In[ ]:


f,ax = plt.subplots(figsize=(8,5))

ax.set_title("Погрешности вычислений, формула центральных разностей")
ax.set_xlabel("h")
ax.set_ylabel("Ошибка")
ax.grid()
ax.plot(hs,np.abs(diffOutputsCent-ftestPrime(x)))


# Можно заметить, что характер погрешности изменился, так как ошибка имеет старший член порядка $h^2$. Давайте проверим экспериментально, с какой скоростью убывает ошибка и сравним два метода.

# In[ ]:


f,ax = plt.subplots(figsize=(8,5))

ax.set_title("Погрешности")
ax.set_xlabel("log h")
ax.set_ylabel("Ошибка(Лог)")
ax.grid()
ax.plot(  np.log(hs)/np.log(10),np.log( np.abs(diffOutputsRight-ftestPrime(x)) )/np.log(10)  )
ax.plot(  np.log(hs)/np.log(10),np.log( np.abs(diffOutputsCent-ftestPrime(x)) )/np.log(10)  )
ax.legend(["правые", "центральные"])


# Из этого графика видно, что по мере уменьшения h, ошибка начинает убывать примерно как старший член погрешности $Ch^k$ и мы можем примерно вычислить $k$ и $C$ для двух приведённых методов.

# ## Что если функция оценена неточно?
# 
# На практике такое происходит очень часто:
# 
# 1. GPS-показания со смартфона обычно очень неточные;
# 2. Экспонометр, использующийся для фотокамеры может выдавать неточный замер;
# 3. Различные датчики (температуры, освещенности...) в устройствах умного дома тоже обычно неточные;
# 4. ...

# Один из способов моделировать подобные ошибки -- предположить, что они случайные и имеют какое-то распределение с компактным носителем. Пусть $\tilde{f}_i$ будут зашумлёнными оценками функции $f$ в точках $x_i$. Предположим, что шум $\epsilon_i = f_i - \tilde{f}_i$ имеет нулевое среднее и почти наверное лежит в отрезке $[-\delta,\delta]$.  
# 
# Это даёт, что для всех $i$ модуль шума $\vert f_i - \tilde{f}_i\vert \leq \delta$.
# 
# Теория говорит, что погрешность формул дифференцирования в этом случае будет вести себя совсем плохо. Проверим на эксперименте, предположив, что шум равномерно распределён в отрезке $[-\delta,\delta]$.

# In[ ]:


delta=1e-8

hs = np.arange(1e-6,0.7,1e-4)
x=0.1
np.random.seed(10540)#для воспроизводимости
noises = np.random.uniform(size=(2,len(hs)))*2*delta-delta


def ftest(x):
    return np.exp(-x**2)
def ftestPrime(x):
    return -2*x*np.exp(-x**2)

diffOutputsRight = np.zeros([len(hs)])
diffOutputsCent = np.zeros([len(hs)])
for k in np.arange(len(hs)):
    #Right
    xs=np.array([x,x+hs[k]])
    differentiator = FinDifferenceDerivative(coefs=np.array([-1,1])/hs[k],xs=np.array([0,1])*hs[k] )
    diffOutputsRight[k] = differentiator.compute(ftest(xs)+noises[:,k])

    #Central
    xs=np.array([x-hs[k],x+hs[k]])
    differentiator = FinDifferenceDerivative(coefs=np.array([-1,1])/(2*hs[k]),\
                                             xs=np.array([-1,1])*hs[k] )                         
    diffOutputsCent[k] = differentiator.compute(ftest(xs)+noises[:,k])


# In[ ]:


f,ax = plt.subplots(figsize=(8,5))

ax.set_title("Погрешности")
ax.set_xlabel("log h")
ax.set_ylabel("Ошибка(Лог)")
ax.grid()
ax.plot(  np.log(hs)/np.log(10),np.log( np.abs(diffOutputsRight-ftestPrime(x)) )/np.log(10)  )
ax.plot(  np.log(hs)/np.log(10),np.log( np.abs(diffOutputsCent-ftestPrime(x)) )/np.log(10)  )


# Видно, что даже совсем малые шумы(сравнимые с погрешностью округления) существенно меняют картину. Если же мы меняем шум до более практических значений, то схемы практически разваливаются.
# 
# Чуть позже мы посмотрим, как можно вычислять производные другими методами, включающими в себя интерполяцию и аппроксимацию, тем самым попытавшись в конкретном приложении сгладить эффект неустойчивости.

# In[ ]:




