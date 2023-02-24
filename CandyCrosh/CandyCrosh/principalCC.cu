
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//Dividir qué funciones se ejecutarán en la GPU y qué funciones en la CPU:

//Elementos:

//* int[?][?] Tablero
//* int numVidas

//Funciones:

//* void generarTablero(int caramelos)		//Generar tablero del juego. Como parámetro de entrada se encuentra el número de caramelos distintos que queremos tener
//* bool comprobarBloques(int[] bloques)			//Se introduce como parámetros un array de arrays. Estos arrays contienen las coordenadas de los bloques que se quieren eliminar (Contemplar tbn combinaciones verticales)
//													//Si sale 'false'. entonces se restará una vida
//* void modificarTablero(int[] coordenadas)			//Dados las coordenadas de los bloques que eliminar como parámetro de entrada, eliminarlos y hacer que el resto de piezas caigan



