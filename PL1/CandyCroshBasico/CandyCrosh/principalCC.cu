﻿#include <curand_kernel.h>
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
__global__ void rellenarTablero(int* tablero, int nFilas, int nColumnas, int tiposN, curandState* state, int seed) {

	int row = threadIdx.x;
	int column = threadIdx.y;

	//Iniciar el generador aleatorio
	curand_init(seed, row * nColumnas + column, 0, &state[row * nColumnas + column]);
	for (int rellenarColumna = 0; rellenarColumna < nColumnas; ++rellenarColumna) {
		if (column < nColumnas && row < nFilas && tablero[row * nColumnas + column] == 0) {
			tablero[row * nColumnas + column] = (curand(&state[row * nColumnas + column]) % tiposN + 1);
		}
	}
}


//Comprueba que el bloque dado permita ser eliminado, y en caso afirmativo, elimina dichos elementos sobrescribiéndolos por 0:
__global__ void eliminarBloques(int* tablero, int nRows, int nColumns, int coordY, int coordX) {

	int row = threadIdx.x;
	int column = threadIdx.y;
	int carameloElegido = tablero[coordY * nColumns + coordX];

	//Los hilos que pertenezcan a la fila de la posicion elegida ejecutan esto
	if (row * column < (nRows * nColumns) && row == coordY) {
		int start = coordX;
		int end = coordX;

		//Mientras haya caramelos iguales antes de nuestra posicion, llevar la posicion de la columna de inicio atras
		while (start > 0 && tablero[row * nColumns + start - 1] == carameloElegido) start--;

		//Mientras haya caramelos iguales despues de nuestra posicion, aumentar la posicion de la columna de fin.
		while (end < nColumns - 1 && tablero[row * nColumns + end + 1] == carameloElegido) end++;

		//Si la diferencia entre inicio y fin es mayor que 2, borramos todos los elementos poniendo un 0
		if (end - start + 1 >= 2) {
			for (int i = start; i <= end; i++) {
				tablero[row * nColumns + i] = 0;
			}
		}
	}
	//Los hilos de la columna de la posicion elegida ejecutan el else:
	else if (row * column < (nRows * nColumns) && column == coordX) {
		int start = coordY;
		int end = coordY;
		//Igual que en el codigo de las filas, pero ahora vamos moviendo el inicio y final por las filas, en vez de las columnas
		while (start > 0 && tablero[(start - 1) * nColumns + column] == carameloElegido) start--;
		while (end < nRows - 1 && tablero[(end + 1) * nColumns + column] == carameloElegido) end++;
		//Remplazamos con 0s igual que en la fila
		if (end - start + 1 >= 2) {
			for (int i = start; i <= end; i++) {
				tablero[i * nColumns + column] = 0;
			}
		}
	}
	__syncthreads();

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
__global__ void activarRompecabezas(int* tablero, int colorBloqueEliminar, int nFilas, int nColumnas, int coordX, int coordY) {
	int idx = threadIdx.x;

	for (int i = idx; i < nFilas * nColumnas; i += blockDim.x) {
		//Comprobamos que el índice se encuentre dentro de los límites de la matriz
		if (i < nFilas * nColumnas) {
			//En caso de que la posición analizada sea igual al bloque que se quiere eliminar, se sobrescribe a 0
			if (tablero[i] == colorBloqueEliminar) {
				tablero[i] = 0;
			}
			tablero[coordY * nColumnas + coordX] = 0;
		}
	}
}


//Eliminar todos los bloques en un radio de 4 elementos obteniendo como centro la posición indicada en las coordenadas ('posXActivar', 'posYActivar'):
__global__ void activarTNT(int* tablero, int posXActivar, int posYActivar, int nFilas, int nColumnas) { //'nColumnas' como parámetro para asegurarse de recorrer y borrar todas las apariciones en la matriz
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int radioExplosion = 4;     //Radio de bloques que afectará la explosión del TNT con respecto del centro, que es la posición introducida como parámetro de entrada

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
//*PONER EN LA MEMORIA QUE TBN SE ME HABIA OCURRIDO HACER QUE SE SUBA EL 0 Y BAJAR UNA LSITA CON EL RESTO DE ELEMENTOS, PERO COMO ES COMPLICADO TRABAJAR CON ARRAYS DINAMICOS, FUE DESCARTADO
__global__ void dejarCaerBloques(int* tablero, int nFilas, int nColumnas) {

	int i = threadIdx.x; // calcula el índice correspondiente en la matriz
	int posColumna = i % nColumnas;

	//Se recorre la columna en busca de algún 0:
	for (int lugarColumna = 0; lugarColumna < nFilas; ++lugarColumna) {
		if (tablero[posColumna + (nColumnas * lugarColumna)] == 0) {
			int posicionBloqueCero = posColumna + (nColumnas * lugarColumna);
			//En caso de encontrar un 0, vamos a iterar hasta que se encuentre en la primera fila de la matriz:
			while ((posicionBloqueCero / nColumnas) > 0) {
				tablero[posicionBloqueCero] = tablero[posicionBloqueCero - nColumnas];
				tablero[posicionBloqueCero - nColumnas] = 0;
				posicionBloqueCero -= nColumnas;
			}
			//Escribimos un 0 en la primera fila de la matriz:
			tablero[posicionBloqueCero] = 0;
		}
	}
}

__device__ int posicionesCero = 0;
__global__ void ponerPowerup(int* tablero, int nFilas, int nColumnas, int coordY, int coordX, int carameloEnPos) {
	int fila = threadIdx.x;
	int columna = threadIdx.y;
	posicionesCero = 0;
	if (tablero[fila * nColumnas + columna] == 0) {
		atomicAdd(&posicionesCero, 1);
	}
	__syncthreads();
	if (posicionesCero == 5) {
		//El 10 es una bomba
		tablero[coordY * nColumnas + coordX] = 10;
	}
	else if (posicionesCero == 6) {
		//El 20 es una TNT
		tablero[coordY * nColumnas + coordX] = 20;
	}
	else if (posicionesCero > 6) {
		//El 5x es un rompecabezas
		tablero[coordY * nColumnas + coordX] = 50 + carameloEnPos % 10;
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
		printf("Input no aceptado. %s", prompt);
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
				if (valorCelda == 1) {
					//Imprimimos el valor del caramelo
					printf(" \033[0;36m%d\033[0m ", valorCelda);
				}
				else if (valorCelda == 2) {
					printf(" \033[0;31m%d\033[0m ", valorCelda);
				}
				else if (valorCelda == 3) {
					printf(" \033[0;33m%d\033[0m ", valorCelda);
				}
				else if (valorCelda == 4) {
					printf(" \033[0;32m%d\033[0m ", valorCelda);
				}
				else if (valorCelda == 5) {
					printf(" \x1b[38;5;130m%d\x1b[0m ", valorCelda);
				}
				else if (valorCelda == 6) {
					printf(" \x1b[38;5;226m%d\x1b[0m ", valorCelda);
				}
			}
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char** argv) {

	int modo = 1;
	int tiposCaramelos = 4;
	int filas = 1;
	int columnas = 1;
	if (argc == 5) {

		char* primer = argv[1];
		if (primer[1] == 'a') {
			modo = 2;
		}
		else if (primer[1] == 'm') {
			modo = 1;
		}
		int dificultad = atoi(argv[2]);
		if (dificultad == 1) {
			tiposCaramelos = 4;
		}
		else if (dificultad == 2) {
			tiposCaramelos = 6;

		}
		filas = atoi(argv[3]);
		columnas = atoi(argv[4]);
	}
	else {
		printf("\n \t\tBIENVENIDO A CUNDY CROSH SOGA! \n");
		printf("----------------------------------------------------------------\n");
		printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
		printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
		printf("\nANTES DE COMENZAR A JUGAR SELECCIONA, SELECCIONA LOS PARAMETROS DEL JUEGO :)\n");
		modo = validate_input("\nEl juego cuenta con 2 modos de ejecucion: \n1.- Manual \n2.- Automatico\n Elige una:");
		int dificultad = validate_input("\nAhora, elige la dificultad: \n1.- Dificil (6 tipos de caramelos) \n2.- Facil (4 tipos de caramelos\n Elige una:");
		if (dificultad == 1) {
			tiposCaramelos = 6;
		}
		else if (dificultad == 2) {
			tiposCaramelos = 4;
		}
		filas = validate_input("\nIntroduce el numero de filas del tablero de juego: ");
		columnas = validate_input("\nFinalmente, introduce el numero de columnas del tablero de juego: ");
	}


	int vidas = 5;


	int* tablero_dev;
	int* tablero_host = (int*)malloc(filas * columnas * sizeof(int));


	//Llenar de 0 la matriz inicial
	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			tablero_host[i * columnas + j] = 0;
		}
	}

	curandState* state;

	//Dar memoria a la matriz y el generador aleatorio en la GPU
	cudaMalloc((void**)&state, filas * columnas * sizeof(curandState));
	cudaMalloc((void**)&tablero_dev, filas * columnas * sizeof(int));

	cudaMemcpy(tablero_dev, tablero_host, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(1);
	dim3 threads(filas, columnas);

	//BUCLE DEL JUEGO!!!
	int coordX;
	int coordY;
	int puntos = 0;

	while (vidas > 0) {
		//Al empezar cada ronda, rellenar el tablero con caramelos
		system("cls");
		printf("\n \t\tCUNDY CROSH SOGA \n");		
		printf("----------------------------------------------------------------\n");
		printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
		printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
		rellenarTablero << < block, threads >> > (tablero_dev, filas, columnas, tiposCaramelos, state, time(NULL));
		cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\n\t\tSCORE %d\n", puntos);
		print_matrix((int*)tablero_host, filas, columnas);
		printf("\t\tVidas restantes: \033[0;31m%d\033[0m\n\n", vidas);

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

		//Intentar eliminar bloques en la posicion que se ha indicado
		int valor = tablero_host[coordY * columnas + coordX];
		if (tablero_host[coordY * filas + coordX] == 10) {
			int filaCol = rand() % 2;
			if (filaCol == 1) {
				activarBomba << <block, threads >> > (tablero_dev, coordY, filaCol, filas, columnas);
			}
			else if (filaCol == 0) {
				activarBomba << <block, threads >> > (tablero_dev, coordX, filaCol, filas, columnas);
			}
		}
		else if (tablero_host[coordY * columnas + coordX] == 20) {
			activarTNT << <block, threads >> > (tablero_dev, coordX, coordY, filas, columnas);
		}
		else if (tablero_host[coordY * columnas + coordX] > 49 && tablero_host[coordY * columnas + coordX] < 57) {
			activarRompecabezas << <block, threads >> > (tablero_dev, tablero_host[coordY * columnas + coordX] % 10, filas, columnas, coordX, coordY);
		}
		else {
			eliminarBloques << < 1, threads >> > (tablero_dev, filas, columnas, coordY, coordX);
			ponerPowerup << < 1, threads >> > (tablero_dev, filas, columnas, coordY, coordX, valor);
		}

		cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);

		int caramelosElim = posicionesEliminadas((int*)tablero_host, filas, columnas);
		if (caramelosElim == 0) {
			//Si no se ha eliminado ningun caramelo con el kernel
			vidas--;
			printf("\nPosicion mala: te quedan %d vidas\n", vidas);
			getchar();
		}
		else {
			//Cuando si se ha modificado el tablero
			puntos += caramelosElim * 10;
			system("cls");
			printf("\n \t\tCUNDY CROSH SOGA\n");
			printf("----------------------------------------------------------------\n");
			printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
			printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
			printf("\n\t\tSCORE %d\n", puntos);
			print_matrix((int*)tablero_host, filas, columnas);
			printf("\t\tVidas restantes: \033[0;31m%d\033[0m\n\n", vidas);
			getchar();
			system("cls");
			printf("\n \t\tCUNDY CROSH SOGA\n");
			printf("----------------------------------------------------------------\n");
			printf("*Paradigmas Avanzados de Programacion, 3GII* 31 de marzo de 2023\n");
			printf("By: Daniel de Heras Zorita y Adrian Borges Cano\n");
			dejarCaerBloques << <block, columnas >> > (tablero_dev, filas, columnas);
			cudaMemcpy(tablero_host, tablero_dev, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
			printf("\n\t\tSCORE %d\n", puntos);
			print_matrix((int*)tablero_host, filas, columnas);
			printf("\t\tVidas restantes: \033[0;31m%d\033[0m\n\n", vidas);
			getchar();
		}
	}

	printf("\n\tGAME OVER :v\n");
	printf("\n\tSCORE FINAL: %d\n", puntos);
	printf("\n\tGracias por jugar!\n");
	printf("\n\tBy: Daniel De Heras y Adrian Borges\n");
	printf("\n\n-------------------------------------------------------\n\n");

	cudaFree(tablero_dev);
	cudaFree(state);

	return 0;
}


