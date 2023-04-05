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
      val comienzoFilaBorrar: Int = (posicionBomba/numColumnas)*numColumnas ; val finFilaBorrar: Int = comienzoFilaBorrar+numColumnas-1
      activarBombaFila(tablero, comienzoFilaBorrar, finFilaBorrar)
    } else { //Eliminamos la columna de la posición
      val comienzoColumnaBorrar: Int = posicionBomba%numColumnas
      activarBombaColumna(tablero, comienzoColumnaBorrar, numColumnas)
    }
  }

  //Sobrescribir a 0 los bloques de la fila indicada por el rango de elementos. Los índices comienzan en 0:
  def activarBombaFila(tablero:List[Int], inicioFilaBorrar:Int, finFilaBorrar:Int):List[Int] ={
    if(inicioFilaBorrar<finFilaBorrar){
      val tableroModificado:List[Int] = reemplazarElemento(tablero, inicioFilaBorrar, 0)
      activarBombaFila(tableroModificado, inicioFilaBorrar+1, finFilaBorrar)
    } else {
      reemplazarElemento(tablero,inicioFilaBorrar,0)
  }}

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
  
  //TODO: activarRompecabezas

  //TODO: activarTNT

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


  }
}