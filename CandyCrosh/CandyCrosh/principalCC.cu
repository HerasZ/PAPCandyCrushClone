
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

//Dividir qué funciones se ejecutarán en la GPU y qué funciones en la CPU:

//Elementos:

int** tablero;
const int numVidas = 3;

//Funciones:

//Generación del tablero, el cual se encarga a la GPU para no sobrecargar la CPU:
__global__ void generarTablero(int* tablero, int nFilas, int nColumnas, int tiposN, curandState* state) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //Iniciar el generador aleatorio
    curand_init(3456, j, 0, &state[j]);
    if (j < nColumnas && i < nFilas) {
        tablero[i * nColumnas + j] = (curand(&state[i * nColumnas + j])%tiposN+1);
    }
}

//Impresión de la matriz por pantalla:
void print_matrix(int* mtx, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", mtx[i*n+j]);
        }
        printf("\n");
    }
}

__global__ void checkAndReplace(int* matrix, int row, int col) {
    const int tamMatriz = 5;
    __shared__ int rowValues[tamMatriz];
    __shared__ int colValues[tamMatriz];
    int i, j;

    // Cada hilo carga los valores de su fila y columna correspondientes en memoria compartida
    i = threadIdx.x;
    j = blockIdx.x;
    rowValues[i] = matrix[row * tamMatriz + i];
    colValues[i] = matrix[i * tamMatriz + col];
    __syncthreads();

    // Cada hilo comprueba si su valor es igual a algún otro valor en su fila o columna
    if (rowValues[i] == colValues[j]) {
        // Si hay coincidencia, se sustituyen los valores por 0
        matrix[row * tamMatriz + i] = 0;
        matrix[i * tamMatriz + col] = 0;
    }
}

//Comprueba que el bloque dado permita ser eliminado, y en caso afirmativo, elimina dichos elementos sobrescribiéndolos por 0:
//* PROBAR QUE FUNCIONE BIEN
__global__ void eliminarBloques(int* tablero, int size, int fila, int columna) {
 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = tid / size;
    int column = tid % size;
    int carameloElegido = tablero[fila * size + columna];

    //Los hilos que pertenezcan a la fila de la posicion elegida ejecutan esto
    if (tid < (size*size) && row == fila) {
        int start = columna;
        int end = columna;
        
        //Mientras haya caramelos iguales antes de nuestra posicion, llevar la posicion de la columna de inicio atras
        while (start > 0 && tablero[row * size + start - 1] == carameloElegido) start--;

        //Mientras haya caramelos iguales despues de nuestra posicion, aumentar la posicion de la columna de fin.
        while (end < size - 1 && tablero[row * size + end + 1] == carameloElegido) end++;

        //Si la diferencia entre inicio y fin es mayor que 2, borramos todos los elementos poniendo un 0
        if (end - start + 1 >= 2)
        {
            for (int i = start; i <= end; i++)
            {
                tablero[row * size + i] = 0;
            }
        }

    }

    //Los hilos de la columna de la posicion elegida ejecutan el else:
    else if (tid < (size*size) && column == columna)
    {
        int start = fila;
        int end = fila;

        //Igual que en el codigo de las filas, pero ahora vamos moviendo el inicio y final por las filas, en vez de las columnas
        while (start > 0 && tablero[(start - 1) * size + column] == carameloElegido) start--;
        while (end < size - 1 && tablero[(end + 1) * size + column] == carameloElegido) end++;

        //Remplazamos con 0s igual que en la fila
        if (end - start + 1 >= 2)
        {
            for (int i = start; i <= end; i++)
            {
                tablero[i * size + column] = 0;
            }
        }
    }
    __syncthreads();
}
//Eliminar el número de la fila o columna indicada por 'posActivar'. Si 'filaColumna' es True, entonces borra la fila, si es False, borra la columna:
__global__ void activarBomba(int* tablero, int posActivar, bool filaColumna, int nFilas, int nColumnas ) {
    int x = threadIdx.x;
//    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (filaColumna) {
        if (posActivar < nFilas) {
            if (x < nFilas) {
                tablero[posActivar * nColumnas + x] = 0;
        }}
        else {
            //De momento lo comento porque se imprime por cada hilo
            //printf("\n\n ERROR: No es posible borrar una fila fuera del rango de la matriz\n\n");
    }}
    else {
        if (posActivar < nColumnas) {
            if (x < nColumnas) {
                tablero[x * nColumnas + posActivar] = 0;
            }} else{
            //printf("\n\n ERROR: No es posible borrar una columna fuera del rango de la matriz\n\n");
    }} 
}

//activarRompecabezas

/*__global__ void activarTNT(int* tablero, int fila, int columna) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int tamannoMatriz = 15;
    tablero[fila * tamannoMatriz + columna] = 0;
    tablero[fila * tamannoMatriz + columna+1] = 0;
    tablero[(fila+1) * tamannoMatriz + columna] = 0;

    
    if (row == 4 || col == 4) {
        return;
    }
    int index_next = row * 5 + col + 1;
    int index_up = (row - 1) * 5 + col;
    matrix[index_next] = 0;
    matrix[index_up] = 0;
}
*/


//-------------------------------------------------------------------------------------------------------------------------------------
// '1' si es fácil(1,2,3,4), '2' si es difícil(1,2,3,4,5,6) + número de filas del tablero + número de columnas del tablero

int main(int argc, char** argv) { 
    const int filas = 15; 
    const int columnas = 15;
    int tiposCaramelos = 6;

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
    cudaMalloc((void**)&tablero_dev,filas*columnas*sizeof(int));

    cudaMemcpy(tablero_dev, tablero_host, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);
    dim3 blocks(filas, columnas);
    dim3 threads(filas, columnas);
    generarTablero<<< blocks,threads >>>(tablero_dev,filas,columnas,tiposCaramelos,state);

    cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);




    printf("\n");
    print_matrix((int*)tablero_host, filas, columnas);
    cudaMemcpy(tablero_dev, tablero_host, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);
    //activarTNT << < blocks, threads >> > (tablero_dev, 2, 2);
    //activarBomba << <blocks, threads >> > (tablero_dev, 2, 1, filas, columnas);
    eliminarBloques << <1, filas*columnas >> > (tablero_dev, filas, 2, 2);
    //checkAndReplace << <blocks, threads >> > (tablero_dev, 2,2);
    cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n");
    print_matrix((int*)tablero_host, filas, columnas);

    return 0;
}





