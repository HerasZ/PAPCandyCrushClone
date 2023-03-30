#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <ctype.h>


//Variables:

int** tablero;

//Funciones:

//Generación del tablero, el cual se encarga a la GPU para no sobrecargar la CPU:
__global__ void rellenarTablero(int* tablero, int nFilas, int nColumnas, int tiposN, curandState* state) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //Iniciar el generador aleatorio
    curand_init(3456, j, 0, &state[j]);
    if (j < nColumnas && i < nFilas && tablero[i * nColumnas + j] == 0) {
        tablero[i * nColumnas + j] = (curand(&state[i * nColumnas + j]) % tiposN + 1);
    }
}


//Comprueba que el bloque dado permita ser eliminado, y en caso afirmativo, elimina dichos elementos sobrescribiéndolos por 0:
__global__ void eliminarBloques(int* tablero, int nRows, int nColumns, int coordY, int coordX) {

    __shared__ int matrizCompartida[10][10];

    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int columna = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < nColumns && columna < nRows) {
        matrizCompartida[fila][columna] = tablero[fila * nRows + columna];
    }

    __syncthreads();

    int carameloElegido = tablero[coordY * nRows + coordX];

    //Los hilos que pertenezcan a la fila de la posicion elegida ejecutan esto
    if (fila < nColumns && columna < nRows && fila == coordY) {
        int start = coordX;
        int end = coordX;

        //Mientras haya caramelos iguales antes de nuestra posicion, llevar la posicion de la columna de inicio atras
        while (start > 0 && matrizCompartida[fila][start-1] == carameloElegido) start--;

        //Mientras haya caramelos iguales despues de nuestra posicion, aumentar la posicion de la columna de fin.
        while (end < nColumns - 1 && matrizCompartida[fila][end + 1] == carameloElegido) end++;

        //Si la diferencia entre inicio y fin es mayor que 2, borramos todos los elementos poniendo un 0
        if (end - start + 1 >= 2) {
            for (int k = start; k <= end; k++) {
                tablero[fila * nRows + k] = 0;
            }
        }
    }

    //Los hilos de la columna de la posicion elegida ejecutan el else:
    if (fila < nColumns && columna < nRows && columna == coordX) {
        int start = coordY;
        int end = coordY;

        //Igual que en el codigo de las filas, pero ahora vamos moviendo el inicio y final por las filas, en vez de las columnas
        while (start > 0 && matrizCompartida[start - 1][columna] == carameloElegido) start--;
        while (end < nRows - 1 && matrizCompartida[end + 1][columna] == carameloElegido) end++;
        //Remplazamos con 0s igual que en la fila
        if (end - start + 1 >= 2) {
            for (int k = start; k <= end; k++) {
                tablero[k * nRows + columna] = 0;
            }
        }
    }
}

//Eliminar el número de la fila o columna indicada por 'posActivar'. Si 'filaColumna' es True, entonces borra la fila, si es False, borra la columna:
__global__ void activarBomba(int* tablero, int posActivar, bool filaColumna, int nFilas, int nColumnas) {
    int x = threadIdx.x;
    //    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (filaColumna) {
        if (posActivar < nFilas) {
            if (x < nFilas) {
                tablero[posActivar * nColumnas + x] = 0;
            }
        }
        else {
            //De momento lo comento porque se imprime por cada hilo
            //printf("\n\n ERROR: No es posible borrar una fila fuera del rango de la matriz\n\n");
        }
    }
    else {
        if (posActivar < nColumnas) {
            if (x < nColumnas) {
                tablero[x * nColumnas + posActivar] = 0;
            }
        }
        else {
            //printf("\n\n ERROR: No es posible borrar una columna fuera del rango de la matriz\n\n");
        }
    }
}

//Eliminar todas las apariciones de un color de caramelo (que corresponde a un número entre 1-6) en el tablero:
__global__ void activarRompecabezas(int* tablero, int colorBloqueEliminar, int nFilas, int nColumnas, int coordX, int coordY) { //'nColumnas' como parámetro para asegurarse de recorrer y borrar todas las apariciones en la matriz
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //Comprobamos que el índice se encuentre dentro de los límites de la matriz
    if (i*j < nFilas * nColumnas) {
        //En caso de que la posición analizada sea igual al bloque que se quiere eliminar, se sobrescribe a 0
        if (tablero[i * nFilas + j] == colorBloqueEliminar) {
            tablero[i * nFilas + j] = 0;
        }
    }
    tablero[coordY * nFilas + coordX] = 0;

}

//Eliminar todos los bloques en un radio de 4 elementos obteniendo como centro la posición indicada en las coordenadas ('posXActivar', 'posYActivar'):
__global__ void activarTNT(int* tablero, int posXActivar, int posYActivar, int nFilas, int nColumnas) { //'nColumnas' como parámetro para asegurarse de recorrer y borrar todas las apariciones en la matriz
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int radioExplosion = 2;     //Radio de bloques que afectará la explosión del TNT con respecto del centro, que es la posición introducida como parámetro de entrada

    //Comprobamos que el índice se encuentre dentro de los límites de la matriz
    if (i < nFilas * nColumnas) {
        //Comprobamos que la posición analizada esté dentro del rango de 'radioExplosion' elementos de radio
        if ((i >= posYActivar - radioExplosion) && (i <= posYActivar + radioExplosion) &&
            (j >= posXActivar - radioExplosion) && (j <= posXActivar + radioExplosion)) {
            tablero[i * nColumnas + j] = 0;
        }
    }
}

//Sobreescribir bloques con valor 0 con el valor de los bloques que se encuentren arriba de este. En caso de no tener bloques por encima, se generarán nuevos bloques:
__global__ void dejarCaerBloques(int* tablero, int nFilas, int nColumnas) {

    // nuevo índice que tiene en cuenta el número de hilos por bloque y bloques
    int i = blockIdx.x;
    int posColumna = threadIdx.x;

    if (i * posColumna < nFilas * nColumnas) {
        //Se recorre la columna en busca de algún 0:
        for (int lugarColumna = 0; lugarColumna < nFilas; ++lugarColumna) {
            if (tablero[posColumna + (nColumnas * lugarColumna)] == 0) {
                int posicionBloqueCero = posColumna + (nColumnas * lugarColumna);
                //En caso de encontrar un 0, vamos a iterar hasta que se encuentre en la primera fila de la matriz:
                while ((posicionBloqueCero) >= posColumna) {
                    //printf("\nHilo %d Cambia su posicion %d por %d\n", i, tablero[posicionBloqueCero], tablero[posicionBloqueCero - nColumnas]);
                    tablero[posicionBloqueCero] = tablero[posicionBloqueCero - nColumnas];
                    tablero[posicionBloqueCero - nColumnas] = 0;
                    posicionBloqueCero -= nColumnas;
                }
            }
        }
    }
}

__device__ int posicionesCero = 0;
__global__ void ponerPowerup(int* tablero, int nFilas, int nColumnas, int coordY, int coordX, int carameloEnPos) {
    int fila = blockIdx.x;
    int columna = threadIdx.x;
    posicionesCero = 0;
    if (tablero[fila * nFilas + columna] == 0) {
        atomicAdd(&posicionesCero, 1);
    }
    __syncthreads();
    if (posicionesCero == 5) {
        //El 10 es una bomba
        tablero[coordY * nFilas + coordX] = 10;
    }
    else if (posicionesCero == 6) {
        //El 20 es una TNT
        tablero[coordY * nFilas + coordX] = 20;
    }
    else if (posicionesCero > 6) {
        //El 5x es un rompecabezas
        tablero[coordY * nFilas + coordX] = 50 + carameloEnPos % 10;
    }
}


//-------------------------------------------------------------------------------------------------------------------------------------
// '1' si es fácil(1,2,3,4), '2' si es difícil(1,2,3,4,5,6) + número de filas del tablero + número de columnas del tablero


int validate_input(const char* prompt) {
    int num;
    char c;

    printf("%s", prompt);
    while (scanf("%d%c", &num, &c) != 2 || c != '\n') {
        while (getchar() != '\n');
        printf("Invalid input. %s", prompt);
    }

    return num;
}

int posicionesEliminadas(int* mtx, int m, int n) {
    int veces = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (mtx[i * n + j] == 0) {
                veces++;
            }

        }
    }
    return veces;
}


//Impresión de la matriz por pantalla:
void print_matrix(int* mtx, int m, int n) {
    printf("\n");
    int valorCelda;
    for (int i = 0; i < m; i++) {
        printf("\t");
        for (int j = 0; j < n; j++) {
            valorCelda = mtx[i * n + j];
            if (valorCelda == 0) {
                //Si el valor es 0 (elemento borrado) no imprimimos nada
                printf("   ");
            }
            else if (valorCelda == 10) {
                //La bomba se representa con B al imprimir
                printf(" B ");
            }
            else if (valorCelda == 20) {
                //La TNT se representa con T al imprimir
                printf(" T ");
            }
            else if (valorCelda > 49 && valorCelda < 57) {
                //El rompecabezas se representa con Rx al imprimir
                printf("R%d ", (valorCelda % 10));
            }
            else {
                //Imprimimos el valor del caramelo
                printf(" %d ", valorCelda);
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    const int filas = 10;
    const int columnas = 10;
    int tiposCaramelos = 4;
    int vidas = 5;
    int modo = 1; //1 manual, 2 automatico

    int* tablero_dev;
    int tablero_host[filas][columnas];

    //Llenar de 0 la matriz inicial
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            tablero_host[i][j] = 0;
        }
    }

    curandState* state;

    //Dar memoria a la matriz y el generador aleatorio en la GPU
    cudaMalloc((void**)&state, filas * columnas * sizeof(curandState));
    cudaMalloc((void**)&tablero_dev, filas * columnas * sizeof(int));

    cudaMemcpy(tablero_dev, tablero_host, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);
    dim3 blocks(filas, columnas);
    dim3 threads(filas, columnas);
    printf("\nGeneracion inicial del tablero:\n");


    //BUCLE DEL JUEGO!!!
    int coordX;
    int coordY;


    while (vidas > 0) {
        //Al empezar cada ronda, rellenar el tablero con caramelos
        system("cls");
        printf("\n \t\tCUNDY CROSH SOGA\n");
        printf("----------------------------------------------------------------\n");
        printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
        printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
        rellenarTablero << < blocks, threads >> > (tablero_dev, filas, columnas, tiposCaramelos, state);
        cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
        print_matrix((int*)tablero_host, filas, columnas);

        //Pedir las coordenadas al usuario
        if (modo == 1) {
            coordY = validate_input("Introduce la coordenada Y (fila): ") - 1;
            coordX = validate_input("Introduce la coordenada X (columna): ") - 1;
        }
        else {
            coordY = rand() % filas;
            coordX = rand() % columnas;
            printf("Posicion elegida aleatoriamente: Fila %d, Columna %d", coordY + 1, coordX + 1);
            getchar();
        }

        int valor = tablero_host[coordY][coordX];

        //Intentar eliminar bloques en la posicion que se ha indicado
        if (tablero_host[coordY][coordX] == 10) {
            bool filaCol = rand() % 2;
            if (filaCol) {
                activarBomba << <blocks, threads >> > (tablero_dev, coordY, filaCol, filas, columnas);
            }
            else if (filaCol) {
                activarBomba << <blocks, threads >> > (tablero_dev, coordX, filaCol, filas, columnas);
            }
        }
        else if (tablero_host[coordY][coordX] == 20) {
            activarTNT << <blocks, threads >> > (tablero_dev, coordX, coordY, filas, columnas);
        }
        else if (tablero_host[coordY][coordX] > 49 && tablero_host[coordY][coordX] < 57) {
            activarRompecabezas << <blocks, threads >> > (tablero_dev, tablero_host[coordY][coordX] % 10, filas, columnas, coordX, coordY);
        }
        else {
            
            eliminarBloques << < blocks, threads >> > (tablero_dev, filas, columnas, coordY, coordX);
            ponerPowerup << <filas, columnas >> > (tablero_dev, filas, columnas, coordY, coordX, valor);
        }

        cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);

        if (posicionesEliminadas((int*)tablero_host, filas, columnas) == 0) {
            //Si no se ha eliminado ningun caramelo con el kernel
            vidas--;
            printf("\nPosicion mala: te quedan %d vidas\n", vidas);
            getchar();
        }
        else {
            //Cuando si se ha modificado el tablero
            system("cls");
            printf("\n \t\tCUNDY CROSH SOGA\n");
            printf("----------------------------------------------------------------\n");
            printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
            printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
            print_matrix((int*)tablero_host, filas, columnas);
            getchar();
            system("cls");
            printf("\n \t\tCUNDY CROSH SOGA\n");
            printf("----------------------------------------------------------------\n");
            printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
            printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
            dejarCaerBloques << <filas, columnas >> > (tablero_dev, filas, columnas);
            cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
            print_matrix((int*)tablero_host, filas, columnas);
            getchar();
        }
    }

    printf("\n\tGAME OVER X_X\n");
    printf("\n\tGracias por jugar!\n");
    printf("\n\tBy: Daniel De Heras y Adrian Borges\n");
    printf("\n\n-------------------------------------------------------\n\n");

    cudaFree(tablero_dev);
    cudaFree(state);

    return 0;
}


