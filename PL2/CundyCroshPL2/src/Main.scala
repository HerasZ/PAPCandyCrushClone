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
      activarBombaFila(tablero, comienzoFilaBorrar, finFilaBorrar, numColumnas)
    } else { //Eliminamos la columna de la posición
      val comienzoColumnaBorrar: Int = posicionBomba%numColumnas
      activarBombaColumna(tablero, comienzoColumnaBorrar, numColumnas)
    }
  }
  //TODO: Hacer 'refactor' a este metodo porque tbn es usado en el TNT
  //Sobrescribir a 0 los bloques de la fila indicada por el rango de elementos. Los índices comienzan en 0:
  def activarBombaFila(tablero:List[Int], inicioFilaBorrar:Int, finFilaBorrar:Int, numeroColumnas:Int):List[Int] ={
    //'numeroColumnas' para poder comprobar que el rango de elementos a eliminar se encuentra en una misma fila de la matriz,
    // y no elimina elementos de otras filas
    if (inicioFilaBorrar>0 && (inicioFilaBorrar%numeroColumnas)<numeroColumnas){
      if (inicioFilaBorrar < finFilaBorrar) {
        val tableroModificado: List[Int] = reemplazarElemento(tablero, inicioFilaBorrar, 0)
        activarBombaFila(tableroModificado, inicioFilaBorrar + 1, finFilaBorrar, numeroColumnas)
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
    if (indiceReemplazar<=0) elementoReemplazar::tableroModificar.tail
    else tableroModificar.head::reemplazarElemento(tableroModificar.tail, indiceReemplazar-1, elementoReemplazar)
  }

  //Devuelve el número de elementos que contiene la lista:
  def longitudLista(lista: List[Int]): Int = lista match {
    //Si la lista está vacía:
    case Nil => 0
    //Si la lista solo tiene un elemento:
    case head :: Nil => 1
    //Si la lista tiene más de un elemento
    case head :: tail => 1 + longitudLista(tail)
  }

  //Borra todos los números iguales al entero 'colorBorrar' dentro de la lista 'tablero':
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

    activarTNT_Aux(tablero, comienzoIterador, finalIterador, comienzoBorrar, finalBorrar, numFilas, numColumnas)

    //TODO adri: LLamar metodo aux
    /*
    if (comienzoIterador<0){ //El comienzo del iterador está fuera de los rangos
      if (finalIterador>numFilas){ //El final del iterador sobrepasa el tamaño de la matriz

      } else { //El final del iterador no sobrepasa el tamaño de la matriz

      }
    } else {  //El comienzo del iterador está dentro de la matriz
      if (finalIterador > numFilas) { //El final del iterador sobrepasa el tamaño de la matriz

      } else {    //El final del iterador tampoco sobrepasa el tamaño de la matriz

      }
    }
    activarTNT_Aux(tablero, inicioIt, finIt, inicioBorrar, finBorar)

    //TODO: Calcular las posiciones a lo ancho de los elementos que se van a borrar en las filas
    //TODO: Pensar si todos los if es mejor meterlos en el Aux




    filaActual:Int = posicionActivar/numColumnas //TODO: Comprobar que es mayor que 0
    finIterador //TODO: Comprobar que es menor que el numColumnas de la matriz

    if (filaBorrar==dentroTablero) {
      activarBombaFila(tableroModificadoConLoAnterior, inicioQueCalcular, finQueCalcular)
    }
    //TODO: La idea es calcular el rango de filas que hay que borrar dependiendo del radio de Explosión que asignemos,
    // y usamos el métoodo 'activarBombaFila' para borrarlas
    */
  }

  def activarTNT_Aux(tablero:List[Int], inicioIterador:Int, finIterador:Int, inicioBorrar:Int, finBorrar:Int, numeroFilas:Int, numeroColumnas:Int):List[Int] = {
    //'inicioIterador' indica la fila por la que se comienza a iterar

    //Si el comienzo del iterador no ha terminado, y va a recorrer una fila dentro del rango de la matriz:
    if (inicioIterador<finIterador && inicioIterador<numeroFilas) {
      //Si el Comienzo del iterador fuera del rango superior
      if (inicioIterador < 0) {
        //Si el inicio del iterador es menor que 0, entonces no haremos nada hasta entrar en los límites de la matriz
        activarTNT_Aux(tablero, inicioIterador + 1, finIterador, inicioBorrar, finBorrar, numeroFilas, numeroColumnas)
      } else if (inicioIterador==0){
        val tableroModificado: List[Int] = activarBombaFila(tablero, inicioBorrar, finBorrar, numeroFilas)
        activarTNT_Aux(tableroModificado, inicioIterador + 1, finIterador, inicioBorrar, finBorrar, numeroFilas, numeroColumnas)
      } else {
        val tableroModificado: List[Int] = activarBombaFila(tablero, inicioBorrar+numeroColumnas, finBorrar+numeroColumnas, numeroFilas)
        activarTNT_Aux(tableroModificado, inicioIterador + 1, finIterador, inicioBorrar, finBorrar, numeroFilas, numeroColumnas)
      }
    } else tablero //El iterador ha llegado a su fin o va a intentar iterar sobre una posición externa a la matriz. Por tanto, se termina el método

    /*
      //Si el Fin del iterador fuera del rango inferior
      if(finIterador > numFilas) {
        val tableroModificado: List[Int] = activarBombaFila(tablero, inicioBorrar, finBorrar)
        activarTNT_Aux(tableroModificado, inicioIterador + 1, finIterador, inicioBorrar, finBorrar)
        BorrarFila
        Volver a llamar a
        LlamadaMetodoconInicio = 0 + finIterador = numeroFilas
      } else {
        BorrarFila
        Volver a llamar a…
        LlamadaMetodoconInicio = 0 + finIteradorNormal
      }
    } else {
      If(finIterador > numeroFilas) {
        BorrarFila
        Volver a llamar a
        LlamadaMetodoconInicioNormal + finIterador = numeroFilas
      }else{
        BorrarFila
        Volver a llamar a
        LlamadaMetodoconInicioNormal + finIteradorNormal
      }
    }
    -si comienzos iterador >= fin iterador
    , devuelve matriz
    -Lo de contemplar que inicioBorrar y finBorrar se encuentren dentro de la matriz se hace dentro de
    ‘activarBombaFila
    InicioBorrar
    FinBorrar









    if (inicioIterador < finIterador) {

      //NO SERÍA NECESARIO PONER TANTOS IF'S SI PUDIESEMOS USAR VARIABLES:

      //Comprobamos si la fila del tablero que se está intentando acceder está dentro del mismo
      if (inicioIterador < 0) { //El comienzo del iterador está fuera de los rangos
        if (inicioIterador > numeroFilas) { //El final del iterador sobrepasa el tamaño de la matriz
          if(inicioBorrar<0) activarTNT_Aux()
        } else { //El final del iterador no sobrepasa el tamaño de la matriz

        }
      } else { //El comienzo del iterador está dentro de la matriz
        if (inicioIterador > numeroFilas) { //El final del iterador sobrepasa el tamaño de la matriz

        } else { //El final del iterador tampoco sobrepasa el tamaño de la matriz

        }
      }

      //TODO: (Empezar desde cero) COSAS QUE METER EN BUCLE AUXILIAR (Empezar desde cero):
      // 1.- Que el comienzo del iterador (eje Y) sea mayor que 0 + que el final de iterador sea menor que numFilas
      // 2.- Que comienzo borrar (eje X) sea mayor que 0 y que el final sea menor que numColumnas
      // 3.- Aplicar un caso base para que el bucle termine



      if (inicioIterador>=0) {    //TODO: Eliminar
          val tableroModificado:List[Int] = activarBombaFila(tablero, inicioBorrar, finBorrar)
          activarTNT_Aux(tableroModificado, inicioIterador+1, finIterador, inicioBorrar, finBorrar)
      //En caso de que dicha fila no esté dentro de los límites, comprobamos si la siguiente fila lo está
      } else {
          activarTNT_Aux(tablero, inicioIterador + 1, finIterador, inicioBorrar, finBorrar)
      }} else tablero
      */
  }

    //TODO: eliminarBloques

    //TODO: elegirBloqueAutomatico

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
   println("\nprobador Bombafila")
   val probador:List[Int] = activarBombaFila(probadorTNT, -4, 9, 8)
   imprimir(probador, 8)
   println("\nactivarTNT")
   val usoTNT: List[Int] = activarTNT(probadorTNT, 27, 8, 8, 4)
   imprimir(usoTNT, 8)
   //TODO: Usar debugger en método de 'activarTNT' para solucionarlo
   }
}