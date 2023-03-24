
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

//Dividir qué funciones se ejecutarán en la GPU y qué funciones en la CPU:

//Elementos:

int** tablero;
const int numVidas;

//Funciones:

//Generación del tablero, el cual se encarga a la GPU para no sobrecargar la CPU:
__global__ void generarTablero(int* tablero, int nFilas, int nColumnas, int tiposN) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < nColumnas && i < nFilas) {
        tablero[i * nColumnas + j] = rand() % tiposN;
    }
}




//* bool comprobarBloques(int[] bloques)			//Se introduce como parámetros un array de arrays. Estos arrays contienen las coordenadas de los bloques que se quieren eliminar (Contemplar tbn combinaciones verticales)
//													//Si sale 'false'. entonces se restará una vida
//Como en C "no existen" booleanos, la función devolverá 1 si es verdadero, y 0 si es falso
// Si es falso, restamos una vida.
int comprobarBloques(int* matrizCoordenadas, int* tablero) {    //Al comprobar las casillas, linealizar la matriz
  //estilo de matrizCoordendas:
  //{2,3}   -> En el caso de fila
  //{2,3}   -> En el caso de columna
    //CONTINUAR
    return 0;
}

//Eliminar los elementos proporcionados en el array de coordenadas:
void eliminarBloques(int* arrayCoordenadas, int tamannoArray) {      //Parámetro de entrada es un array donde cada 2 posiciones guarda unas coordenadas + longitud del array
    if (tamannoArray%2!=0) {     //Tratamiento de errores. Si el array tiene una longitud impar, entonces quiere decir que las coordenadas están incompletas
        printf("\n\n\nERROR. Longitud del array para 'eliminarBloques' debe ser par.\n\n\n");
    } else {
        for (int i = 0; i < tamannoArray; i+=2) {
            tablero[arrayCoordenadas[i]][arrayCoordenadas[i + 1]] = 0;
        }
    }
}
 
//*PREGUNTAR PROFE?* -> Por qué me dice que numcColumnasTablero es una constante?

//Activar la bomba en las coordenadas seleccionadas. La bomba borra todos los elementos de la fila/columan seleccionada
void activarBomba(int filaColumna, int* coordenadasBomba) {      //'1' si queremos borrar fila, '0' si queremos borrar columna + numero de fila/columna que queramos borrar + coordenadas en las que se encuentra la bomba
    int longitudArrayCoordenadas = sizeof(coordenadasBomba) / sizeof(coordenadasBomba[0]);
    if (longitudArrayCoordenadas != 2) {    // || (tablero[coordenadasBomba[0]][coordenadasBomba[1]]!='B' //Comprobamos si las coordenadas están bien introducidas
        printf("\n\n\nERROR. Las coordenadas de la bomba solo pueden ser 2 números que indiquen: Fila, Columna.\n\n\n");
    }
    else {
        if (filaColumna) {      //Si es '1', entonces quiere decir que es 'true'
            int filaBorrar[numColumnasTablero * 2];       //Los elementos que vamos a borrar estarán esparcidos en la fila de la bomba y dependerán del número de columnas que haya. Se multiplica *2 porque son pares de coordenadas
            for (int i = 0; i < numColumnasTablero;i++) {
                filaBorrar[i * 2] = coordenadasBomba[0];
                filaBorrar[i * 2 + 1] = i;
            }
            eliminarBloques(filaBorrar, numColumnasTablero * 2);
        }
        else {                  //Si es '0', entonces quiere decir que es 'false'
            int columnaBorrar[numFilasTablero * 2];       //Los elementos que vamos a borrar estarán esparcidos en la fila de la bomba y dependerán del número de columnas que haya. Se multiplica *2 porque son pares de coordenadas
            for (int i = 0; i < numFilasTablero;i++) {
                columnaBorrar[i * 2] = coordenadasBomba[0];
                columnaBorrar[i * 2 + 1] = i;
            }
            eliminarBloques(columnaBorrar, numColumnasTablero * 2);
        }
    }
}

//Activar el TNT localizado en las coordenadas seleccionadas. El TNT elimina todos los bloques que se encuentran en un radio de 4 elementos.
void activarTNT(int* coordenadasTNT) {
    int longitudArrayCoordenadas = sizeof(coordenadasTNT) / sizeof(coordenadasTNT[0]);
    if (longitudArrayCoordenadas != 2) {    // || (tablero[coordenadasTNT[0]][coordenadasTNT[1]]!='T' //Comprobamos si las coordenadas están bien introducidas
        printf("\n\n\nERROR. Las coordenadas del TNT solo pueden ser 2 números que indiquen: Fila, Columna.\n\n\n");
    } else {
        //CONTINUAR 
    }

}


//* void activarTNT()


void activarRompecabezas(int* coordenadasRompecabezas) {
    int longitudArrayCoordenadas = sizeof(coordenadasRompecabezas) / sizeof(coordenadasRompecabezas[0]);
    if (longitudArrayCoordenadas != 2) {    // || (tablero[coordenadasTNT[0]][coordenadasTNT[1]]!='T' //Comprobamos si las coordenadas están bien introducidas
        printf("\n\n\nERROR. Las coordenadas del Rompecabezas solo pueden ser 2 números que indiquen: Fila, Columna.\n\n\n");
    }
    else {
        //TODO: Obtener el elemento que se encuentra en las coordenadas introducidas (Ej: R1,R2,...) y de ahí, obtener el color (número)
        int colorBorrar = X;
        //Recorremos secuencialmente el tablero buscando los bloques que sean del mismo color (mismo número) que el del rompecabezas
        for (int i = 0; i < numFilasTablero; ++i) {
            for (int j = 0; i < numColumnasTablero; ++j) {
                if () {     //Si el elemento del tablero es igual al color que borrar, lo eliminamos
                    int coordenadas[2] = { i,j };
                    eliminarBloques(coordenadas, 2);
                }
            }
        }
        //CONTINUAR 
    }
}

//* void dejarCaerBloques(int* tablero)         //Dejamos caer los bloques hasta que las celdas donde tengan un '0' sean rellenadas





//Imprimir tablero del juego:
void imprimirTablero() {
    printf("\n\tTABLERO:\n");
    for (int i = 0; i < numFilasTablero; i++) {
        for (int j = 0; j < numColumnasTablero; j++) {
            printf("%d ", tablero[i][j]);
        }
        printf("\n");
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------
// '1' si es fácil(1,2,3,4), '2' si es difícil(1,2,3,4,5,6) + número de filas del tablero + número de columnas del tablero

int main(int argc, char** argv) { 
    const int filas = (int) argv[3];
    const int columnas = (int)argv[4];
    int tiposCaramelos;
    int bloquesBorrar[9] = {5,4,9,10,8,0,7,9,3};
    int longitudArray = sizeof(bloquesBorrar)/sizeof(bloquesBorrar[0]);

    int* tablero_dev;
    int** tablero_host;

    //Alocar tablero con memoria dinamica
    tablero_host = (int**)malloc(filas * sizeof(int*));
    for (int i = 0; i < filas; i++) {
        tablero_host[i] = (int*)malloc(columnas * sizeof(int));
    }

    printf("Longitud del array: %d ", longitudArray);
    generarTablero<<< 1,1 >>>(tablero_dev,filas,columnas,tiposCaramelos);
    eliminarBloques((int*)bloquesBorrar, longitudArray);
    imprimirTablero();

    for (int i = 0; i < filas; i++) {
        free(tablero_host[i]);
    }
    free(tablero_host);

    return 0;
}





