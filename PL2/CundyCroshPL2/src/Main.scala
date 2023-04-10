import scala.util.Random
object Main {

  val numFilas: Int = 0
  val numColumnas: Int = 0
  val random = new Random()

  //Sustituir en los elementos que valgan 0 un número aleatorio entre los posibles indicados:
  def rellenarTablero(posibilidadesBloques:Int, tableroRellenar:List[Int]):List[Int] = {
    tableroRellenar match {
      //En caso de que la lista no tenga elementos:
      case Nil => Nil
      //En caso de que la lista tenga al menos un elemento:
      case _ => if (tableroRellenar.head==0) (random.nextInt(posibilidadesBloques) + 1)::rellenarTablero(posibilidadesBloques, tableroRellenar.tail)
                        else rellenarTablero(posibilidadesBloques, tableroRellenar.tail)
    }
  }

  //Al seleccionar en una determinada casilla, eliminar todos los elementos que se encuentren en la fila/columna de la posición indicada:
  def activarBomba(tablero:List[Int], posicionBomba:Int, numColumnas:Int, filaColumna:Boolean ):List[Int] ={ //'True' si se elimina fila, 'False' si se elimina columna
    if (filaColumna) { //Eliminamos la fila de la posición
      //Calculamos cual es la primera y última posición de la fila que queremos borrar, para poder eliminarla
      val comienzoFilaBorrar: Int = (posicionBomba/numColumnas)*numColumnas ; val finFilaBorrar: Int = comienzoFilaBorrar+numColumnas-1
      //Llamamos a la función que se encarga de eliminar los bloques del rango indicado
      eliminarElementosFila(tablero, comienzoFilaBorrar, finFilaBorrar, numColumnas)
    } else { //Eliminamos la columna de la posición
      val comienzoColumnaBorrar: Int = posicionBomba%numColumnas
      activarBombaColumna(tablero, comienzoColumnaBorrar, numColumnas)
    }
  }

  //Sobrescribir a 0 los bloques de la fila indicada por el rango de elementos. Los índices comienzan en 0:
  def eliminarElementosFila(tablero:List[Int], inicioFilaBorrar:Int, finFilaBorrar:Int, numeroColumnas:Int):List[Int] ={
    //'numeroColumnas' para poder comprobar que el rango de elementos a eliminar se encuentra en una misma fila de la matriz,
    // y no elimina elementos de otras filas
    if (inicioFilaBorrar>=0){
      if (inicioFilaBorrar < finFilaBorrar) {
        val tableroModificado: List[Int] = reemplazarElemento(tablero, inicioFilaBorrar, 0)
        eliminarElementosFila(tableroModificado, inicioFilaBorrar + 1, finFilaBorrar, numeroColumnas)
      } else {
        reemplazarElemento(tablero, inicioFilaBorrar, 0)
      }} else tablero
    }

  //Sobrescribir a 0 los bloques de la columna indicada en 'columnaBorrar'. El índice comienza en 0:
  def activarBombaColumna(tablero:List[Int], columnaBorrar:Int, numColumnas:Int):List[Int] = {
    if (columnaBorrar<longitudLista(tablero)){
      val tableroModificado:List[Int] = reemplazarElemento(tablero, columnaBorrar, 0)
      activarBombaColumna(tableroModificado, columnaBorrar+numColumnas, numColumnas)
    } else {
      tablero
    }
  }

  //Reemplazar la posición 'indiceReemplazar' del tablero por 'elementoReemplazar'. El índice comienza en 0:
  def reemplazarElemento(tableroModificar: List[Int], indiceReemplazar: Int, elementoReemplazar: Int):List[Int] = {
    if (tableroModificar==Nil) Nil
    else if (indiceReemplazar<=0) elementoReemplazar::tableroModificar.tail
    else tableroModificar.head::reemplazarElemento(tableroModificar.tail, indiceReemplazar-1, elementoReemplazar)
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
  def activarRompecabezas(tablero:List[Int], colorBorrar:Int):List[Int] = {
    tablero match{
      //Si la lista está vacía:
      case Nil => Nil
      //Si la lista contiene un elemento:
      case head::Nil => if (head==colorBorrar) 0::Nil
                        else head::Nil
      //Si la lista contiene más de un elemento:
      case head::tail => if (head == colorBorrar) 0 :: activarRompecabezas(tail,colorBorrar)
                        else head :: activarRompecabezas(tail,colorBorrar)
    }
  }

  def activarTNT(tablero:List[Int], posicionActivar:Int, numFilas:Int, numColumnas:Int, radioExplosion:Int):List[Int] = {
    //Indicamos donde debería comenzar y terminar las filas que se quieren borrar
    val comienzoIterador:Int = (posicionActivar/numColumnas)-radioExplosion
    val finalIterador:Int = (posicionActivar/numColumnas)+radioExplosion
    //Indicamos donde deberia comenzar y terminar el rango de elementos que se quiere borrar de cada fila
    val comienzoBorrar:Int = (posicionActivar%numColumnas)-radioExplosion
    val finalBorrar:Int = (posicionActivar%numColumnas)+radioExplosion

    //Si tanto el comienzo como el final del rango de elementos a borrar en la fila están fuera del tablero:
    if (comienzoBorrar<0 && finalBorrar>=numColumnas) activarTNT_Aux(tablero, comienzoIterador, finalIterador, 0, numColumnas-1, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si solo el comienzo del rango de elementos a borrar en la fila está fuera del tablero:
    else if (comienzoBorrar<0) activarTNT_Aux(tablero, comienzoIterador, finalIterador, 0, finalBorrar, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si solo el final del rango de elementos a borrar en la fila está fuera del tablero:
    else if (finalBorrar>=numColumnas) activarTNT_Aux(tablero, comienzoIterador, finalIterador, comienzoBorrar, numColumnas-1, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
    //Si el rango de elementos a borrar en la fila está dentro del tablero:
    else activarTNT_Aux(tablero, comienzoIterador, finalIterador, comienzoBorrar, finalBorrar, numFilas, numColumnas, comienzoIterador) //El 'rastreador' necesita se inicializado con el mismo valor que el 'comienzoIterador'
  }

  //Función auxiliar que permite recorrer la matriz eliminando las filas afectadas por el TNT.
  // 'rastreador' es una variable auxiliar inicializada con el mismo valor que 'comienzoIterador'.
  // 'rastreador' es usada para comprobar cuándo puede comenzar a eliminar los valores de las filas en caso de que 'comienzoIterador' sea mayor que 0.
  def activarTNT_Aux(tablero:List[Int], inicioIterador:Int, finIterador:Int, inicioBorrar:Int, finBorrar:Int, numeroFilas:Int, numeroColumnas:Int, rastreador:Int):List[Int] = {
    //'inicioIterador' indica la fila por la que se comienza a iterar
    //Si el comienzo del iterador no ha terminado, y va a recorrer una fila dentro del rango de la matriz:
    if (inicioIterador<=finIterador && inicioIterador<numeroFilas) {
      //Si el Comienzo del iterador fuera del rango superior
      if (inicioIterador < 0) {
        //Si el inicio del iterador es menor que 0, entonces no haremos nada hasta entrar en los límites de la matriz
        activarTNT_Aux(tablero, inicioIterador + 1, finIterador, inicioBorrar, finBorrar, numeroFilas, numeroColumnas, rastreador+1)
      } else {
        if (rastreador<=0) {
          val tableroModificado: List[Int] = eliminarElementosFila(tablero, inicioBorrar, finBorrar, numeroFilas)
          activarTNT_Aux(tableroModificado, inicioIterador + 1, finIterador, inicioBorrar + numeroColumnas, finBorrar + numeroColumnas, numeroFilas, numeroColumnas,0)
        } else {
          activarTNT_Aux(tablero, inicioIterador, finIterador, inicioBorrar + numeroColumnas, finBorrar + numeroColumnas, numeroFilas, numeroColumnas, rastreador-1)
        }
      }} else tablero //El iterador ha llegado a su fin o va a intentar iterar sobre una posición externa a la matriz. Por tanto, se termina el método
  }

    //TODO: eliminarBloques

    //TODO: Hacer caer bloques de arriba hacia abajo (utilizar getColumna)

    //TODO: elegirBloqueAutomatico

  //Imprimir la matriz: TODO Adaptarlo para los potenciadores y que los 0 se muestren como vacío
  def imprimir(l: List[Int], numColumnas:Int): Unit = {
    print(" " + l.head + " ")
    if (l.tail.length % numColumnas == 0) {
      print("\n")
      if (l.tail.length > 0) imprimir(l.tail, numColumnas)
    } else if (l.tail.length <= 0) {
      throw new Error("ERROR")
    } else {
      imprimir(l.tail, numColumnas)
    }
  }

 def main(args: Array[String]): Unit = {

    //Impresión de funciones para comprobar que va bien. Al terminar de desarrollarlo al completo, borrarlo
    val listaPrueba = List(0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)
    println("Impresión de lista con ceros\n")
    imprimir(listaPrueba, 5)
    println("Rellenar Tablero\n")
    val nuevaLista:List[Int] = rellenarTablero(6, listaPrueba)
    println(nuevaLista)
    imprimir(nuevaLista,5)
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
    val probadorTNT:List[Int] = List(
     1, 2, 5, 7, 7, 2, 2, 2,
     6, 3, 9, 8, 2, 5, 1, 1,
     4, 7, 9, 1, 8, 4, 9, 7,
     2, 9, 5, 5, 7, 9, 4, 2,
     4, 3, 5, 9, 9, 3, 4, 1,
     5, 9, 7, 3, 7, 5, 8, 2,
     6, 4, 4, 1, 5, 4, 8, 7,
     2, 3, 4, 2, 3, 8, 8, 4)
   println(probadorTNT)
   println("\nmatriz probadorTNT")
   imprimir(probadorTNT, 8)
   println("\nprobador BombaFila")
   val probador:List[Int] = eliminarElementosFila(probadorTNT, -4, 9, 8)
   imprimir(probador, 8)
   println("\nactivarTNT")
   val usoTNT: List[Int] = activarTNT(probadorTNT, 63, 8, 8, 3)   //Fila 4 columna 4 (5)
   imprimir(usoTNT, 8)
   }
}