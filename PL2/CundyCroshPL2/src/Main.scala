import scala.annotation.tailrec
import scala.util.Random
import scala.io.StdIn.readLine

object Main {

  val random = new Random()


  def invertirLista(lista: List[Int]): List[Int] = lista match {
    case Nil => Nil
    case head :: tail => invertirLista(tail) :+ head
  }

  //Sustituir en los elementos que valgan 0 un número aleatorio entre los posibles indicados:
  def rellenarTablero(posibilidadesBloques: Int, tableroRellenar: List[Int]): List[Int] = {
    tableroRellenar match {
      //En caso de que la lista no tenga elementos:
      case Nil => Nil
      //En caso de que la lista tenga al menos un elemento:
      case _ => if (tableroRellenar.head == 0) (random.nextInt(posibilidadesBloques) + 1) :: rellenarTablero(posibilidadesBloques, tableroRellenar.tail)
      else tableroRellenar.head :: rellenarTablero(posibilidadesBloques, tableroRellenar.tail)
    }
  }

  //Al seleccionar en una determinada casilla, eliminar todos los elementos que se encuentren en la fila/columna de la posición indicada:
  def activarBomba(tablero: List[Int], posicionBomba: Int, numColumnas: Int, filaColumna: Boolean): List[Int] = { //'True' si se elimina fila, 'False' si se elimina columna
    if (filaColumna) { //Eliminamos la fila de la posición
      //Calculamos cual es la primera y última posición de la fila que queremos borrar, para poder eliminarla
      val comienzoFilaBorrar: Int = (posicionBomba / numColumnas) * numColumnas
      val finFilaBorrar: Int = comienzoFilaBorrar + numColumnas - 1
      //Llamamos a la función que se encarga de eliminar los bloques del rango indicado
      eliminarElementosFila(tablero, comienzoFilaBorrar, finFilaBorrar, numColumnas)
    } else { //Eliminamos la columna de la posición
      val comienzoColumnaBorrar: Int = posicionBomba % numColumnas
      activarBombaColumna(tablero, comienzoColumnaBorrar, numColumnas)
    }
  }

  //Sobrescribir a 0 los bloques de la fila indicada por el rango de elementos. Los índices comienzan en 0:
  @tailrec
  def eliminarElementosFila(tablero: List[Int], inicioFilaBorrar: Int, finFilaBorrar: Int, numeroColumnas: Int): List[Int] = {
    //'numeroColumnas' para poder comprobar que el rango de elementos a eliminar se encuentra en una misma fila de la matriz,
    // y no elimina elementos de otras filas
    if (inicioFilaBorrar >= 0) {
      if (inicioFilaBorrar < finFilaBorrar) {
        val tableroModificado: List[Int] = reemplazarElemento(tablero, inicioFilaBorrar, 0)
        eliminarElementosFila(tableroModificado, inicioFilaBorrar + 1, finFilaBorrar, numeroColumnas)
      } else {
        reemplazarElemento(tablero, inicioFilaBorrar, 0)
      }
    } else tablero
  }

  //Sobrescribir a 0 los bloques de la columna indicada en 'columnaBorrar'. El índice comienza en 0:
  @tailrec
  def activarBombaColumna(tablero: List[Int], columnaBorrar: Int, numColumnas: Int): List[Int] = {
    if (columnaBorrar < longitudLista(tablero)) {
      val tableroModificado: List[Int] = reemplazarElemento(tablero, columnaBorrar, 0)
      activarBombaColumna(tableroModificado, columnaBorrar + numColumnas, numColumnas)
    } else {
      tablero
    }
  }

  //Reemplazar la posición 'indiceReemplazar' del tablero por 'elementoReemplazar'. El índice comienza en 0:
  def reemplazarElemento(tableroModificar: List[Int], indiceReemplazar: Int, elementoReemplazar: Int): List[Int] = {
    if (tableroModificar == Nil) Nil
    else if (indiceReemplazar <= 0) elementoReemplazar :: tableroModificar.tail
    else tableroModificar.head :: reemplazarElemento(tableroModificar.tail, indiceReemplazar - 1, elementoReemplazar)
  }

  //Devolver el número de elementos que contiene la lista:
  def longitudLista(lista: List[Int]): Int = lista match {
    //Si la lista está vacía:
    case Nil => 0
    //Si la lista solo tiene un elemento:
    case head :: Nil => 1
    //Si la lista tiene más de un elemento
    case head :: tail => 1 + longitudLista(tail)
  }

  //Borrar todos los números iguales al entero 'colorBorrar' dentro de la lista 'tablero':
  def activarRompecabezas(tablero: List[Int], colorBorrar: Int): List[Int] = {
    tablero match {
      //Si la lista está vacía:
      case Nil => Nil
      //Si la lista contiene un elemento:
      case head :: Nil => if (head == colorBorrar) 0 :: Nil
      else head :: Nil
      //Si la lista contiene más de un elemento:
      case head :: tail => if (head == colorBorrar) 0 :: activarRompecabezas(tail, colorBorrar)
      else head :: activarRompecabezas(tail, colorBorrar)
    }
  }

  def activarTNT(tablero: List[Int], posicionActivar: Int, numFilas: Int, numColumnas: Int, radioExplosion: Int): List[Int] = {
    //Indicamos donde debería comenzar y terminar las filas que se quieren borrar
    val comienzoIterador: Int = (posicionActivar / numColumnas) - radioExplosion
    val finalIterador: Int = (posicionActivar / numColumnas) + radioExplosion
    //Indicamos donde deberia comenzar y terminar el rango de elementos que se quiere borrar de cada fila
    val comienzoBorrar: Int = (posicionActivar % numColumnas) - radioExplosion
    val finalBorrar: Int = (posicionActivar % numColumnas) + radioExplosion

    //Si tanto el comienzo como el final del rango de elementos a borrar en la fila están fuera del tablero:
    if (comienzoBorrar < 0 && finalBorrar >= numColumnas) activarTNT_Aux(tablero, comienzoIterador, finalIterador, 0, numColumnas - 1, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si solo el comienzo del rango de elementos a borrar en la fila está fuera del tablero:
    else if (comienzoBorrar < 0) activarTNT_Aux(tablero, comienzoIterador, finalIterador, 0, finalBorrar, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si solo el final del rango de elementos a borrar en la fila está fuera del tablero:
    else if (finalBorrar >= numColumnas) activarTNT_Aux(tablero, comienzoIterador, finalIterador, comienzoBorrar, numColumnas - 1, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si el rango de elementos a borrar en la fila está dentro del tablero:
    else activarTNT_Aux(tablero, comienzoIterador, finalIterador, comienzoBorrar, finalBorrar, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
  }

  //Función auxiliar que permite recorrer la matriz eliminando las filas afectadas por el TNT.
  // 'rastreador' es una variable auxiliar inicializada con el mismo valor que 'comienzoIterador'.
  // 'rastreador' es usada para comprobar cuándo puede comenzar a eliminar los valores de las filas en caso de que 'comienzoIterador' sea mayor que 0.
  @tailrec
  def activarTNT_Aux(tablero: List[Int], inicioIterador: Int, finIterador: Int, inicioBorrar: Int, finBorrar: Int, numeroFilas: Int, numeroColumnas: Int, rastreador: Int): List[Int] = {
    //'inicioIterador' indica la fila por la que se comienza a iterar
    //Si el comienzo del iterador no ha terminado, y va a recorrer una fila dentro del rango de la matriz:
    if (inicioIterador <= finIterador && inicioIterador < numeroFilas) {
      //Si el Comienzo del iterador fuera del rango superior
      if (inicioIterador < 0) {
        //Si el inicio del iterador es menor que 0, entonces no haremos nada hasta entrar en los límites de la matriz
        activarTNT_Aux(tablero, inicioIterador + 1, finIterador, inicioBorrar, finBorrar, numeroFilas, numeroColumnas, rastreador + 1)
      } else {
        if (rastreador <= 0) {
          val tableroModificado: List[Int] = eliminarElementosFila(tablero, inicioBorrar, finBorrar, numeroFilas)
          activarTNT_Aux(tableroModificado, inicioIterador + 1, finIterador, inicioBorrar + numeroColumnas, finBorrar + numeroColumnas, numeroFilas, numeroColumnas, 0)
        } else {
          activarTNT_Aux(tablero, inicioIterador, finIterador, inicioBorrar + numeroColumnas, finBorrar + numeroColumnas, numeroFilas, numeroColumnas, rastreador - 1)
        }
      }
    } else tablero //El iterador ha llegado a su fin o va a intentar iterar sobre una posición externa a la matriz. Por tanto, se termina el método
  }


  //Funcion que se llamara en el main, que se encargara de llamar a su auxiliar para hacer la eliminacion de los bloques
  // y despues comprobara cuantos bloques se han eliminado para poner el power-up correspondiente
  def eliminarBloques(tablero: List[Int], columna: Int, fila: Int, caramelo: Int, numFilas: Int, numColumnas: Int): List[Int] = {
    //Borrar los caramelos contiguos
    val nuevoTablero = eliminarBloques_aux(tablero, columna, fila, caramelo, numFilas, numColumnas)
    //Contar cuantos caramelos hemos borrado
    val caramelosElim = contarEliminados(nuevoTablero)
    caramelosElim match {
      //Si hemos borrado 1 caramelo, devolvemos el tablero tal y como estaba
      case caramelosElim if (caramelosElim <= 1) => tablero
      //Si hemos borrado suficientes caramelos, sustituimos la posicion inicial por el power-up correspondiente
      case caramelosElim if (caramelosElim == 5) => reemplazarElemento(nuevoTablero, fila * numColumnas + columna, 10)
      case caramelosElim if (caramelosElim == 6) => reemplazarElemento(nuevoTablero, fila * numColumnas + columna, 20)
      case caramelosElim if (caramelosElim >= 7) => reemplazarElemento(nuevoTablero, fila * numColumnas + columna, 30 + tablero(fila * numColumnas + columna))
      //Si hemos borrado mas de 1 pero menos de los necesarios para power-ups, devolvemos el tablero con los cambios
      case _ => nuevoTablero
    }
  }

  def eliminarBloques_aux(tablero: List[Int], columna: Int, fila: Int, caramelo: Int, numFilas: Int, numColumnas: Int): List[Int] = {
    //Comprobar que vamos a eliminar en una posicion valida y que es el mismo tipo de caramelo
    if (columna >= 0 && columna < numColumnas && fila >= 0 && fila < numFilas && tablero(fila * numColumnas + columna).equals(caramelo)) {
      val tableroTemp = reemplazarElemento(tablero, fila * numColumnas + columna, 0)
      val tablero1 = eliminarBloques_aux(tableroTemp, columna + 1, fila, caramelo, numFilas, numColumnas)
      val tablero2 = eliminarBloques_aux(tablero1, columna - 1, fila, caramelo, numFilas, numColumnas)
      val tablero3 = eliminarBloques_aux(tablero2, columna, fila + 1, caramelo, numFilas, numColumnas)
      //La llamada con la ultima posicion devuelve el tablero modificado
      eliminarBloques_aux(tablero3, columna, fila - 1, caramelo, numFilas, numColumnas)
    } else {
      //Si nos hemos salido de los limites, devolvemos el tablero sin cambios
      tablero
    }
  }

  def contarEliminados(tablero: List[Int]): Int = tablero match {
    case Nil => 0
    case head :: tail => if (head == 0) 1 + contarEliminados(tail) else contarEliminados(tail)
  }

  //Funcion que llamaremos para hacer subir los ceros de las columnas
  @tailrec
  def flotarCeros(matrix: List[Int], numRows: Int, numCols: Int, currentCol: Int = 0): List[Int] = {
    if (currentCol >= numCols) {
      matrix
    } else {
      // Obtener los indices de todos los 0 en la columna, del de mas arriba al de mas abajo
      val zeroIndices = findCeroIndices(matrix, currentCol, numCols)
      // Mover todos los ceros hacia arriba
      val newMatrix = moverCeros(matrix, zeroIndices, currentCol, numRows, numCols)
      // Pasar a la siguiente columna
      flotarCeros(newMatrix, numRows, numCols, currentCol + 1)
    }
  }

  //Funcion que obtendra todos los indices de los elementos que sean 0 en una columna, devolviendolos en una lista
  def findCeroIndices(matrix: List[Int], currentPos: Int, numCols: Int): List[Int] = {
    //Si nos salimos de la matriz, paramos
    if (currentPos > longitudLista(matrix) - 1) Nil
    else {
      if (matrix(currentPos) == 0) {
        //Si encontramos un 0, ponemos el indice del elemento como cabeza de la lista, y comprobamos el siguiente elemento de la columna
        currentPos :: findCeroIndices(matrix, currentPos + numCols, numCols)
      } else {
        //Si no hay un 0 en la posicion, continuamos sin ponerlo como cabeza.
        findCeroIndices(matrix, currentPos + numCols, numCols)
      }
    }
  }

  //Funcion para mover los 0 de las posiciones de zeroIndices a la parte de arriba de las columnas
  @tailrec
  def moverCeros(matrix: List[Int], zeroIndices: List[Int], currentCol: Int, numRows: Int, numCols: Int): List[Int] = {
    zeroIndices match {
      //Cuando quede la lista vacia
      case Nil => matrix
      //Cuando aun hay elementos por subir
      case head :: tail =>
        //Bajar todos los elementos de la columna
        val newMatrix: List[Int] = bajarColumna(matrix, head, numCols)
        //Llamada recursiva con el resto de posiciones que tienen 0s
        //En la matriz se sustituye
        moverCeros(newMatrix, tail, currentCol, numRows, numCols)

    }
  }

  @tailrec
  def bajarColumna(matrix: List[Int], indexCero: Int, numCols: Int): List[Int] = {
    if (indexCero < numCols) {
      matrix
    } else if (matrix(indexCero - numCols) == 0) {
      matrix
    } else {
      val temp = matrix(indexCero - numCols)
      val cambioMatrix = reemplazarElemento(matrix, indexCero, temp)
      bajarColumna(reemplazarElemento(cambioMatrix, indexCero - numCols, 0), indexCero - numCols, numCols)
    }
  }

  //Elegir automaticamente un bloque al azar del tablero:
  def elegirBloqueAutomatico(tablero: List[Int]): Int = {
    random.nextInt(longitudLista(tablero)) + 1
  }
  //TODO: elegirBloqueAutomaticoOptimizado (Crear nuevo proyecto para ello)

  //Imprimir la matriz: TODO Adaptarlo para los potenciadores
  @tailrec
  def imprimir(l: List[Int], numColumnas: Int): Unit = {
    val elem = l.head
    elem match {
      case 0 => print("   ")
      case 10 => print(" B ")
      case 20 => print(" T ")
      case elem if (elem > 30) => print(" R" + elem / 10 + " ")
      case _ => print(" " + elem + " ")
    }
    if (longitudLista(l.tail) % numColumnas == 0) {
      print("\n")
      if (longitudLista(l.tail) > 0) imprimir(l.tail, numColumnas)
    } else if (longitudLista(l.tail) <= 0) {
      throw new Error("ERROR")
    } else {
      imprimir(l.tail, numColumnas)
    }
  }

  def pedirNumero(maxValue: Int, prompt: String): Int = {
    var num = -1 // initialize num to an invalid value to enter the loop
    while (num < 0 || num > maxValue) {
      print(prompt)
      try {
        num = readLine().toInt
        if (num < 0 || num > maxValue) {
          println(s"Number must be between 0 and $maxValue.")
        }
      } catch {
        case _: NumberFormatException => println("Invalid input. Please enter a number.")
      }
    }
    num
  }

  @tailrec
  def bucleJuego(matriz: List[Int], numColumnas: Int, numFilas: Int, caramelosTipos: Int, vidas: Int): Unit = {
    if (vidas == 0) println("FIN DEL JUEGO") else {
      println()
      val matrizRellena: List[Int] = rellenarTablero(caramelosTipos, matriz)

      println("---" * numColumnas)
      imprimir(matrizRellena, numColumnas)
      println("---" * numColumnas)

      val fila: Int = pedirNumero(numFilas, "Introduce la fila: ") - 1
      val columna: Int = pedirNumero(numFilas, "Introduce la columna: ") - 1
      val carameloElegido: Int = matrizRellena(fila * numColumnas + columna)
      val matrizActualizada: List[Int] = eliminarBloques(matrizRellena, columna, fila, carameloElegido, numFilas, numColumnas)

      val casillasElim: Int = contarEliminados(matrizActualizada)
      if (casillasElim <= 0) {
        println(s"Fallo: Te quedan ${vidas - 1} vidas")
        readLine()
        bucleJuego(matrizActualizada, numColumnas, numFilas, caramelosTipos, vidas - 1)
      } else {

        println("---" * numColumnas)
        imprimir(matrizActualizada, numColumnas)
        println("---" * numColumnas)
        readLine()

        val matrizCaerBloques: List[Int] = flotarCeros(matrizActualizada, numFilas, numColumnas)

        println("---" * numColumnas)
        imprimir(matrizCaerBloques, numColumnas)
        println("---" * numColumnas)
        readLine()

        bucleJuego(matrizCaerBloques, numColumnas, numFilas, caramelosTipos, vidas)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val numFilas = 8
    val numColumnas = 8
    /*
    //Impresión de funciones para comprobar que va bien. Al terminar de desarrollarlo al completo, borrarlo
    val listaPrueba = List(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    println("Impresión de lista con ceros\n")
    imprimir(listaPrueba, 5)
    println("Rellenar Tablero\n")
    val nuevaLista: List[Int] = rellenarTablero(6, listaPrueba)
    println(nuevaLista)
    imprimir(nuevaLista, 5)
    println("Activar Bomba Fila\n")
    val nuevaListaFila: List[Int] = activarBomba(nuevaLista, 7, 5, true)
    println(nuevaListaFila)
    imprimir(nuevaListaFila, 5)
    println("Activar Bomba Columna\n")
    val nuevaListaColumna: List[Int] = activarBomba(nuevaLista, 7, 5, false)
    println(nuevaListaColumna)
    imprimir(nuevaListaColumna, 5)
    println("Activar Rompecabezas\n")
    val nuevaListaRompecabezas: List[Int] = activarRompecabezas(nuevaLista, 3)
    println(nuevaListaRompecabezas)
    imprimir(nuevaListaRompecabezas, 5)
    val probadorTNT: List[Int] = List(
      1, 1, 5, 7, 7, 2, 2, 2,
      6, 1, 4, 8, 2, 5, 1, 1,
      4, 1, 1, 1, 8, 4, 9, 7,
      2, 9, 1, 1, 7, 9, 4, 2,
      4, 3, 5, 1, 9, 3, 4, 1,
      5, 9, 7, 1, 7, 5, 8, 2,
      6, 4, 4, 1, 5, 4, 8, 7,
      2, 3, 4, 6, 3, 8, 8, 4)
    println(probadorTNT)

    //Borrar elementos?
    val borrarCaram: List[Int] = eliminarBloques(probadorTNT, 0, 0, 1)
    println("\nProbando eliminar bloques en fila 0, columna 0")
    imprimir(borrarCaram, 8)

    val caer: List[Int] = flotarCeros(borrarCaram, 8, 8, 0)
    println()
    imprimir(caer, 8)

    println("\nmatriz probadorTNT")
    imprimir(probadorTNT, 8)
    println("\nprobador BombaFila")
    val probador: List[Int] = eliminarElementosFila(probadorTNT, -4, 9, 8)
    imprimir(probador, 8)
    println("\nactivarTNT")
    val usoTNT: List[Int] = activarTNT(probadorTNT, 63, 8, 8, 3) //Fila 4 columna 4 (5)
    imprimir(usoTNT, 8)
    println("\nProbadorElegirBloqueAleatorio")
    println(elegirBloqueAutomatico(probadorTNT))
*/
    val matriz: List[Int] = List.fill(numFilas * numColumnas)(0)
    bucleJuego(matriz, numColumnas, numFilas, 6, 5)

  }

}