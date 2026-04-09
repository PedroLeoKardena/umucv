from abc import ABC, abstractmethod

class MetodoClasificacion(ABC):
    
    @abstractmethod
    def precomputar_modelo(self, nombre, imagen):
        pass

    @abstractmethod
    def clasificar(self, frame):
        pass