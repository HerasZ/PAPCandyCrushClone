import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

import scala.jdk.javaapi.CollectionConverters;

import static java.lang.Thread.sleep;

public class ventanaMatriz extends JFrame implements ActionListener {

    private Image iconoVentana = new ImageIcon("src/Imagenes/appIconCandyCrush.png").getImage();
    private int filas = 10;

    private int columnas = 10;

    private JTable tablaCaramelos;

    private DefaultTableModel modeloCaramelos; //Para modificar la tabla

    private int modoJuego;
    private int numVidas=5;
    private JLabel numVidasLabel = new JLabel();
    private JLabel modoAutomaticoLabel = new JLabel();
    private JLabel filaElegidaLabel = new JLabel();
    private JLabel logoCandyCrushLabel = new JLabel();
    private JLabel columnaElegidaLabel = new JLabel();
    private int dificultad = 6;


    scala.collection.immutable.List<Object> matrizScala;
    private JPanel ventanaMatriz;

    public ventanaMatriz(int modo, int dific, int rows, int column) {
        this.filas = rows;
        this.columnas = column;
        this.modoJuego = modo;
        if (dific == 1) {
            this.dificultad = 4;
        } else if (dific == 2) {
            this.dificultad = 6;
        }

        // Establecer las propiedades de la ventana
        setTitle("Juego - Cundy Crosh");
        setSize(650, 800);
        setLocationRelativeTo(null); // Centrar la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Crear un panel para colocar los componentes
        ventanaMatriz.setLayout(null);
        // Creamos el ImageIcon
        ImageIcon imagenFondo = new ImageIcon("src/Imagenes/fondoPantallaCandyCrushOscuroVertical.png");
        // Cargamos la imagen y creamos un objeto JLabel con la imagen
        JLabel labelFondo = new JLabel(imagenFondo);
        // Agregamos el JLabel al panel y establecemos su posición para ponerlo en el fondo
        labelFondo.setBounds(0, 0, imagenFondo.getIconWidth(), imagenFondo.getIconHeight());
        ventanaMatriz.add(labelFondo);
        //Cambiamos el icono de la ventana
        setIconImage(iconoVentana);
        // Hacemos que la ventana no sea redimensionable
        setResizable(false);

        List<Object> list = new ArrayList<Object>();
        for (int i = 0; i < filas * columnas; i++) {
            list.add(0);
        }

        //scala.collection.immutable.List<Object> result = Main.bucleJuego(CollectionConverters.asScala(list).toList(),8,8,6,5,1,0,0);
        matrizScala = Main.rellenarTablero(dificultad, CollectionConverters.asScala(list).toList());

        modeloCaramelos = new DefaultTableModel() {
            //Hacer las celdas no editables
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        tablaCaramelos = new JTable(modeloCaramelos);
        // Create a couple of columns
        for (int i = 0; i < columnas; i++) {
            modeloCaramelos.addColumn(Integer.toString(i));
        }
        for (int i = 0; i < filas; i++) {
            Object[] row = new Object[columnas];
            for (int j = 0; j < columnas; j++) {
                row[j] = matrizScala.apply(i * columnas + j);
            }
            modeloCaramelos.addRow(row);
        }

        for (int i = 0; i < modeloCaramelos.getColumnCount(); i++) {
            //tablaCaramelos.getColumnModel().getColumn(i).setCellRenderer(new ColumnColorRenderer());
            tablaCaramelos.getColumnModel().getColumn(i).setCellRenderer(new ImageRenderer());
            //Eliminamos las lineas que unen las celdas de la tabla
            tablaCaramelos.setShowGrid(false);
        }

        tablaCaramelos.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
            // do some actions here, for example
            // print first column value from selected row       TODO: Esto es justo lo que tenemos que quitar
            if (numVidas>0){
                int row = 0;
                int col = 0;
                if (modo == 1) {
                    row = tablaCaramelos.rowAtPoint(evt.getPoint());
                    col = tablaCaramelos.columnAtPoint(evt.getPoint());
                } else if (modo == 2) {
                    int pos = Main.getMejorOpcion(matrizScala, filas, columnas, 0, -1, 0);
                    row = pos / columnas;
                    col = pos % columnas;
                    modoAutomaticoLabel.setText("[Modo Automático]");
                    //Imprimimos la fila y columna seleccionada como números naturales. O sea, la primera fila y columna es 0
                    filaElegidaLabel.setText("Fila Elegida: "+ (row+1));
                    columnaElegidaLabel.setText("Columna Elegida: "+(col+1));
                }

                System.out.println(row + " " + col);
                scala.collection.immutable.List<Object> result =
                        Main.bucleJuego(matrizScala, columnas, filas, dificultad, numVidas, 1, row, col);

                if (result.equals(matrizScala)) {
                    System.out.println("Fallo");
                    numVidas--;
                    numVidasLabel.setText("Número de Vidas: "+ numVidas);
                    if (numVidas==0){
                        JOptionPane.showMessageDialog(null, "Te quedaste sin vidas X.X, ¡Gracias por jugar!", "FIN DEL JUEGO", JOptionPane.ERROR_MESSAGE);
                        System.exit(0);
                    }
                } else {
                    matrizScala = result;
                    matrizScala = Main.flotarCeros(matrizScala, filas, columnas, 0);
                    matrizScala = Main.rellenarTablero(dificultad, matrizScala);
                    actualizarMatriz();
                }
            }
        }});
        //Creamos el panel que contendrá el tablero, que estará dentro del panel de la ventana
        JPanel panelTablero = new JPanel();
        ventanaMatriz.add(panelTablero);
        panelTablero.setLayout(null);
        panelTablero.setBounds(70,30,500,500);
        tablaCaramelos.setVisible(true);
        tablaCaramelos.setRowHeight(panelTablero.getHeight()/filas);
        tablaCaramelos.setBounds(0,0,panelTablero.getBounds().width,panelTablero.getBounds().height);
        panelTablero.add(tablaCaramelos);


        //Añadimos el número de vidas
        numVidasLabel.setText("Número de Vidas: " + numVidas);
        Dimension tamannoTexto = numVidasLabel.getPreferredSize();
        numVidasLabel.setFont(new Font("Tahoma", Font.BOLD, 30));
        numVidasLabel.setForeground(Color.WHITE);
        numVidasLabel.setBounds(160, 700, tamannoTexto.width+200, tamannoTexto.height+10);
        if (modo==1){
            //Establecemos las medidas que debe tener el logo para asegurar que no se verá distorsionado
            int anchoLogo = 156;
            int altoLogo = 111;
            //Añadimos el Label del logo de Candy Crush que aparece en el modo manual
            ImageIcon img = new ImageIcon(new ImageIcon("src/Imagenes/logoCandyCrush.png").getImage().getScaledInstance(anchoLogo,altoLogo,Image.SCALE_SMOOTH));
            logoCandyCrushLabel.setIcon(img);
            //Redimensionamos la imagen
            logoCandyCrushLabel.setBounds(240, 560, anchoLogo, altoLogo);
            logoCandyCrushLabel.setSize(anchoLogo,altoLogo);
            labelFondo.add(logoCandyCrushLabel);
        } else if (modo==2){
            //Añadimos los Labels que aparecen cuando estamos en el modo automático
            modoAutomaticoLabel.setFont(new Font("Tahoma", Font.BOLD, 23));
            modoAutomaticoLabel.setForeground(Color.WHITE);
            modoAutomaticoLabel.setBounds(200, 560, tamannoTexto.width+200, tamannoTexto.height+15);
            filaElegidaLabel.setFont(new Font("Tahoma", Font.PLAIN, 18));
            filaElegidaLabel.setForeground(Color.WHITE);
            filaElegidaLabel.setBounds(230, 610, tamannoTexto.width+100, tamannoTexto.height+5);
            columnaElegidaLabel.setFont(new Font("Tahoma", Font.PLAIN, 18));
            columnaElegidaLabel.setForeground(Color.WHITE);
            columnaElegidaLabel.setBounds(230, 640, tamannoTexto.width+100, tamannoTexto.height);
        }
        //Añadimos una pequeña referencia que verifique nuestra autoría en el trabajo
        JLabel autoresLabel = new JLabel("By: DHZ y ABC");
        autoresLabel.setFont(new Font("Tahoma", Font.PLAIN, 10));
        autoresLabel.setForeground(Color.WHITE);
        autoresLabel.setBounds(5, 745, tamannoTexto.width, tamannoTexto.height);
        //Insertamos todos los Labels en el panel
        labelFondo.add(numVidasLabel);
        labelFondo.add(modoAutomaticoLabel);
        labelFondo.add(filaElegidaLabel);
        labelFondo.add(columnaElegidaLabel);
        labelFondo.add(autoresLabel);

        // Agregamos el panel a la ventana
        this.add(ventanaMatriz);
        this.setVisible(true);
    }

    private void actualizarMatriz() {
        modeloCaramelos.setRowCount(0);

        for (int i = 0; i < filas; i++) {
            Object[] row = new Object[columnas];
            for (int j = 0; j < columnas; j++) {
                row[j] = matrizScala.apply(i * columnas + j);
            }
            modeloCaramelos.addRow(row);
        }
        modeloCaramelos.fireTableDataChanged();
    }

    @Override
    public void actionPerformed(ActionEvent e) {}

    //TODO: Si no lo usamos, borrarlo
    class ColumnColorRenderer extends DefaultTableCellRenderer {
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
            Component cell = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
            if (value == null) {
                return cell;
            }
            int tempValue = (int) value;
            if (tempValue > 30) {
                cell.setForeground(Color.white);
                cell.setBackground(Color.darkGray);
            } else if (tempValue == 0) {
                cell.setBackground(Color.white);
                cell.setForeground(Color.white);
            } else if (tempValue == 1) {
                cell.setBackground(Color.cyan);
                cell.setForeground(Color.cyan);
            } else if (tempValue == 2) {
                cell.setBackground(Color.red);
                cell.setForeground(Color.red);
            } else if (tempValue == 3) {
                cell.setBackground(Color.orange);
                cell.setForeground(Color.orange);
            } else if (tempValue == 4) {
                cell.setBackground(Color.green);
                cell.setForeground(Color.green);
            } else if (tempValue == 5) {
                cell.setBackground(new Color(150, 75, 0));
                cell.setForeground(new Color(150, 75, 0));
            } else if (tempValue == 6) {
                cell.setBackground(Color.yellow);
                cell.setForeground(Color.yellow);
            } else if (tempValue == 10) {
                cell.setBackground(Color.black);
                cell.setForeground(Color.black);
            } else if (tempValue == 20) {
                cell.setBackground(Color.pink);
                cell.setBackground(Color.pink);
            }
            return cell;
        }
    }

    //Para hacer que las celdas tengan una imagen
    public class ImageRenderer extends DefaultTableCellRenderer {

        private ImageIcon carameloAzul;
        private ImageIcon carameloRojo;
        private ImageIcon carameloNaranja;
        private ImageIcon carameloVerde;
        private ImageIcon carameloMarronMorado;
        private ImageIcon carameloAmarillo;

        private ImageIcon bomba;
        private ImageIcon rompecabezas;
        private ImageIcon TNT;
        private ImageIcon carameloVacio;


        public ImageRenderer() {
            // Cargamos las imágenes
            carameloAzul = new ImageIcon("src/Imagenes/carameloAzul.png");
            carameloRojo = new ImageIcon("src/Imagenes/carameloRojo.png");
            carameloNaranja = new ImageIcon("src/Imagenes/carameloNaranja.png");
            carameloVerde = new ImageIcon("src/Imagenes/carameloVerde.png");
            carameloMarronMorado = new ImageIcon("src/Imagenes/carameloMarronMorado.png");
            carameloAmarillo = new ImageIcon("src/Imagenes/carameloAmarillo.png");
            bomba = new ImageIcon("src/Imagenes/bomba.png");
            rompecabezas = new ImageIcon("src/Imagenes/rompecabezas.png");
            TNT = new ImageIcon("src/Imagenes/tnt.png");
            carameloVacio = new ImageIcon("src/Imagenes/carameloVacio.png");
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value,
                                                       boolean isSelected, boolean hasFocus, int row, int column) {
            Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

            // Convertimos el valor de la celda a un número entero
            int number = Integer.parseInt(value.toString());

            ImageIcon imagenCelda;
            c.setForeground(Color.white);

            // Asignamos la imagen correspondiente en función de si es par o impar
            if (number == 1) {
                imagenCelda = carameloAzul;
            } else if (number == 2){
                imagenCelda = carameloRojo;
            } else if (number == 3){
            imagenCelda = carameloNaranja;
            } else if (number == 4){
                imagenCelda = carameloVerde;
            } else if (number == 5){
                imagenCelda = carameloMarronMorado;
            } else if (number == 6){
                imagenCelda = carameloAmarillo;
            } else if (number == 10){
                imagenCelda = bomba;
            } else if (number == 20){
                imagenCelda = TNT;
            } else if (number > 30){
                imagenCelda = rompecabezas;
            } else {  //El valor es cero
                imagenCelda = carameloVacio;
            }

            // Ajustamos el tamaño de la imagen a la altura y ancho de la celda
            if (table.getRowHeight(row) > 0) {
                int imageHeight = imagenCelda.getIconHeight();
                int cellHeight = table.getRowHeight(row);
                int imageWidth = imagenCelda.getIconWidth();
                int cellWidth = table.getColumnModel().getColumn(column).getWidth();

                if (imageHeight > cellHeight) {
                    imageWidth = (int) Math.round((double) imageWidth * ((double) cellHeight / (double) imageHeight));
                    imageHeight = cellHeight;
                }
                if (imageWidth > cellWidth) {
                    imageHeight = (int) Math.round((double) imageHeight * ((double) cellWidth / (double) imageWidth));
                    imageWidth = cellWidth;
                }
                setIcon(new ImageIcon(imagenCelda.getImage().getScaledInstance(imageWidth, imageHeight, java.awt.Image.SCALE_SMOOTH)));
            }
            return c;
        }
    }

    //TODO: Borrar cuando terminemos de debuggear
    public static void main(String[] args) {
        ventanaMatriz ventana1 = new ventanaMatriz(1,2,10,10);
        ventana1.setVisible(true);
    }

}