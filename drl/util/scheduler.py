from abc import ABC, abstractmethod

class Schduler(ABC):
    @abstractmethod
    def value(self, t: float) -> float:
        pass
    
    def __call__(self, t: float) -> float:
        return self.value(t)
    
class ConstantScheduler(Schduler):
    def __init__(self, value: float):
        self._value = value
        
    def value(self, t: float) -> float:
        return self._value
    
class LinearScheduler(Schduler):
    def __init__(self, t0: float, t1: float, y0: float, y1: float):
        self._slope = (y1 - y0) / (t1 - t0)
        self._intercept = y0 - self._slope * t0
        
        self._t0 = t0
        self._t1 = t1
        self._y0 = y0
        self._y1 = y1
        
    def value(self, t: float) -> float:
        if t <= self._t0:
            val = self._y0
        elif t >= self._t1:
            val = self._y1
        else:
            val = self._slope * t + self._intercept
        return val
        