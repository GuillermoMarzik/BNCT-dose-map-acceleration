import numpy as np
import re
from typing import  Optional

class Tally:
    """clase para leer los outputs y procesar los tallies
    
    -el formato del archivo mctal esta en en el manual del mcnp6 la-cp-13-00634, sección 6.3.4
    -la forma de unir tallies de varias corridas está en LA-UR-08-0249_tally_merge.pdf
     dentro de los manuales del mcnp5

    """    
    def __init__(self) -> None:
        self._tallies={}
        self._datosCorrida={}
        return

    def limpiar(self) -> None:
        self._tallies={}
        self._datosCorrida={}
        return

    

    def get_tallies(self):
        return self._tallies

    tallies=property(fget=get_tallies,fset=None)

    def get_datosCorrida(self):
        return self._datosCorrida
       
    @staticmethod
    def save_tally(tally: dict, output_dir: str, filename: str):
        """
        Guarda los tallies (con la cantidad de canales que tengan [VER ESTO MÁS ADELANTE]) en el directorio especificado en formato npy.
        """
        with open(output_dir+filename,'wb') as f:
            for i in tally.keys(): # guardo cada canal por separado
                np.save(f, tally[i]['valores'][...,0])
                np.save(f, tally[i]['valores'][...,1])
    
    @staticmethod
    def load_tally(filename):
        """
        Carga un archivo npy con los datos de distintas tallies y las guarda en una matriz de numpy stackeada.
        Si el mapa es de baja calidad, el formato es:
            Canal 0 de esa matriz va a ser la primera tally, canal 1 va a ser el error de esa primera tally, canal 2 va a ser la segunda
            tally, canal 3 el error de esa tally y así sucesivamente.
        Si es de alta calidad (ground truth) el formato es:
            Canal 0 primera tally, canal 1 segunda tally, etc.

        El formato del array de salida es:
            (x,y,z,ch)
        siendo ch el canal que estemos viendo.
        
        Args:
        -----
        filename: str
        	Nombre del archivo
        Returns:
        --------
        tal_stack: ndarray
        	Array de numpy con los valores de las tallies y de sus errores (si aplica)
        """
        all_read = False
        l_tal = []
        with open(filename,'rb') as f:
            while not all_read:
                try:
                    l_tal.append(np.load(f))
                except:
                    all_read = True
        #tal_stack = np.stack((l_tal[0],l_tal[1]),axis=-1)
        #for i in range(2,len(l_tal)):
        #    tal_stack = np.concatenate((tal_stack,np.expand_dims(l_tal[i],axis=-1)),axis=-1)
        #TEST MUCHAS TALLIES
        tal_stack = np.concatenate((l_tal[0],l_tal[1]),axis=-1)
        for i in range(2,len(l_tal)):
            tal_stack = np.concatenate((tal_stack, l_tal[i]), axis=-1)
        return tal_stack

