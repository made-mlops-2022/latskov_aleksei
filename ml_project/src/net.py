# module net.py

import numpy as np


class Module(object):
    """
    Модуль обрабатывает входные данные и выдает выходные.
    Он умеет делать прямой проход и обратный
    
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Вычисляет выход модуля по входным данным
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Вычисляет шаг обратного распространиния ошибки
        
        Включает в себя:
          - вычисление градиента относительно 'input' (нужен для дальнейшего обратного распространения),
          - вычисление градиента относительно параметры (для обновления параметров при оптимизации).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Вычисляет вывод, используя текущий набор параметров класса и ввода.
        """ 
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Вычисление градиента.
        Так же обновляется соответствующая переменная 'gradInput'
        """
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Вычисление градиента относительно его собственных параметров.
        Не переопределяет, если модуль не имеет параметров.
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Обнуляет переменную 'gradParams', если модуль имеет параметры.
        """
        pass
        
    def getParameters(self):
        """
        Возвращает список с его параметрами.
        Если у модуля нет параметров, вернет пустой список.
        """
        return []
        
    def getGradParameters(self):
        """
        Возвращает список с градиентами относительно его параметров.
        Если у модуля нет параметров, вернет пустой список.
        """
        return []
    
    def train(self):
        """
        Устанавливает режим обучения для модуля.
        Поведение обучения и тестирования отличается для Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Устанавливает режим оценки для модуля.
        Поведение обучения и тестирования отличается для Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Удобочитаемое описание.
        """
        return "Module"

    
class Sequential(Module):
    """
        Этот класс реализует контейнер, который последовательно обрабатывает входные данные.
        'input' обрабатывается каждым модулем (слоем) в self.modules последовательно.
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Добовляет модуль в контейнер.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Базовый прямой проход.
        """
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
            
        return self.output

    def backward(self, input, gradOutput):
        """
        Базовый обратный проход.
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   
        """
        for i in range(len(self.modules) - 1, 0, -1):
            gradOutput = self.modules[i].backward(self.modules[i - 1].output, gradOutput)
        self.gradInput = self.modules[0].backward(input, gradOutput)
        
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Собирает все параметры в список.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Собирает все градиенты по параметрам в список.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Распространяет параметр обучения по всем модулям
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Распространяет параметр оценка по всем модулям
        """
        self.training = False
        for module in self.modules:
            module.evaluate()
            

class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
        Вычисляет функцию потерь, связанную с критерием.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
        Вычисляет градиенты функции потерь, связанной с критерием.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Функция, которую нужно переопределить.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Функция, которую нужно переопределить.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Удобочитаемое описание.
        """
        return "Criterion"
    
    
class Linear(Module):
    """
    Модуль (полносвязный слой), выполняющий линейное преобразование.
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = input @ self.W.T + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput @ self.W
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = np.dot(gradOutput.T, input)
        self.gradb = np.sum(gradOutput, axis=0)
        
        
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q
    
    
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # нормализация
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)
        
        return self.output
    
    def updateGradInput(self, _input, gradOutput):
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"
    
    
class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # нормализация
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        self.output -= np.log(np.sum(np.exp(self.output), axis = 1, keepdims=True))
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        new_grad_output = gradOutput / np.exp(self.output)
        conv = np.sum( np.exp(self.output) * new_grad_output, axis=1, keepdims=True)
        self.gradInput = np.exp(self.output) * (new_grad_output - conv)
        
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"
    
    
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, _input):
        self.output = np.maximum(_input, 0)
        return self.output
    
    def updateGradInput(self, _input, gradOutput):
        self.gradInput = np.multiply(gradOutput , _input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"
    
    
class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.zeros_like(input)
        self.output[input >= 0] = input[input >= 0]
        self.output[input < 0] = input[input < 0] * self.slope
        
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0) + np.multiply(gradOutput , input <= 0) * self.slope
        
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"
    
    
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        if self.training:
            self.mask = np.random.choice([True, False], size=input.shape, p=[1 - self.p, self.p])
            self.output = np.copy(input)
            self.output[~self.mask] = 0.
            np.divide(self.output, (1. - self.p), out=self.output)
        else:
            self.output = input
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        grad = np.ones(input.shape)
        grad[~self.mask] = 0
        self.gradInput = gradOutput * grad / (1. - self.p)
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"
    
    
class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        
    def updateOutput(self, input):
        if self.moving_mean is None:
            self.moving_mean = 0
        if self.moving_variance is None:
            self.moving_variance = 0
            
        if self.training:
            b_mean = np.mean(input, axis=0, keepdims = True)
            b_var = np.var(input, axis=0, keepdims = True)
            self.output = (input - b_mean) / np.sqrt(b_var + self.EPS)
            self.moving_mean = self.moving_mean * self.alpha + b_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + b_var * (1 - self.alpha)
        else:
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        var_sqrt = np.sqrt(np.var(input, axis=0, keepdims = True) + self.EPS)
        m = np.mean(gradOutput * self.output, axis=0, keepdims = True)
        self.gradInput = (gradOutput  - self.output * m - gradOutput.mean(axis=0)) / var_sqrt
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"
    
    
class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        self.output = -np.sum(target * input) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"
    

def simple_sgd(variables, gradients, config, state): 
    # Простая структура для храниния накапливаемых значений градиентов
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            current_var -= config['learning_rate'] * current_grad
            var_index += 1   


def adam_optimizer(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            # update `current_var_first_moment`, `var_second_moment` and `current_var` values
            #np.add(... , out=var_first_moment)
            #np.add(... , out=var_second_moment)
            #current_var -= ...
            np.add(config['beta1'] * var_first_moment,
                   (1 - config['beta1']) * current_grad, 
                   out=var_first_moment)
            np.add(config['beta2'] * var_second_moment,
                   (1 - config['beta2']) * current_grad ** 2,
                   out=var_second_moment)
            np.subtract(
                current_var,
                lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon']),
                out=current_var
            )
            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1


def get_batches(dataset, batch_size):
    # batch generator
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]
        
        
def get_model(n_in, n_out, n_sine=4):
    net = Sequential()
    net.add(Linear(n_in, n_in * n_sine))
    net.add(BatchNormalization())
    net.add(Dropout(0.3))
    net.add(ReLU())
    net.add(Linear(n_in * n_sine, n_in * n_sine))
    net.add(BatchNormalization())
    net.add(Dropout(0.2))
    net.add(ReLU())
    net.add(Linear(n_in * n_sine, n_out))
    net.add(BatchNormalization())
    net.add(Dropout(0.1))
    net.add(LogSoftMax())
    
    return net
