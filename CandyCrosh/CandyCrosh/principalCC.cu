
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//Dividir qué funciones se ejecutarán en la GPU y qué funciones en la CPU:

//Elementos:

//* int[?][?] Tablero
//* int numVidas

//Funciones:

//* bool comprobarBloques(int[] bloques)			//Se introduce como parámetros un array de arrays. Estos arrays contienen las coordenadas de los bloques que se quieren eliminar (Contemplar tbn combinaciones verticales)
//													//Si sale 'false'. entonces se restará una vida
//Como en C "no existen" booleanos, la función devolverá 1 si es verdadero, y 0 si es falso
int comprobarBloques(int* matrizCoordenadas, int* tablero) {    //Al comprobar las casillas, linealizar la matriz
  //estilo de matrizCoordendas:
  //{{2,3}, {2,4}, {2,5}}   -> En el caso de fila
  //{{2,3}, {3,3}, {4,4}}   -> En el caso de columna
  

}


//* void eliminarBloques(int[] coordenadas)			//Dados las coordenadas de los bloques que eliminar como parámetro de entrada, eliminarlos. Tbn tener en cuenta que se debe poner B,T,R dependiendo de la combinación
// realizada. Poner un '0' en los bloques que se hayan eliminado, para luego en el método 'dejarCaerBloques' saber dónde tienen que caer
//* void activarBomba(int nFilaBorrar)          //Sobrecargamos el método para que la bomba borre la fila o la columna con la que se ha hecho la combinación
//* void activarBomba(int nColumnaBorrar)       
//* void activarTNT()
//* void activarRompecabezas(int bloqueEliminar)
//* void dejarCaerBloques(int* tablero)         //Dejamos caer los bloques hasta que las celdas donde tengan un '0' sean rellenadas


//TODO: Plantear cómo hacer para que el tablero sea una variable del main, y esta sea escrita con el método 'generarTablero' (Contemplar que lo mandamos a la GPU)
//TODO: Importaciones para 'srand'. Si 'srand' se puede quitar, entonces tbn quitamos dichos 'include'
#include <stdlib.h>
#include <time.h>
//Método de generación del tablero, el cual se encarga a la GPU para no sobrecargar la CPU:
__global__ void generarTablero(int bloques, int nFilas, int nColumnas) {
  int i, j;
  int matriz[nFilas][nColumnas];

  //TODO: Si la quitamos sigue funcionando?
  srand(time(NULL)); // Inicializar la semilla para generar números aleatorios

  // Llenar la matriz con números aleatorios del 1 al 6
  for (i = 0; i < filas; i++) {
    for (j = 0; j < columnas; j++) {
      matriz[i][j] = rand() % bloques + 1;
    }
  }

  // Mostrar la matriz generada
  printf("Matriz generada:\n");
  for (i = 0; i < filas; i++) {
    for (j = 0; j < columnas; j++) {
      printf("%d ", matriz[i][j]);
    }
    printf("\n");
  }
}

//* void imprimirTablero()      // (?) Parámetro de entrada el tablero o no hace falta?

//-------------------------------------------------------------------------------------------------------------------------------------

int main() {
  int filas=12;
  int columnas=12;
  int tiposCaramelos = 6;

  printf("Número de filas: %d", filas);
  printf("Número de columnas: %d", columnas);

  generarMatriz(tiposCaramelos, filas, columnas);

  return 0;
}





