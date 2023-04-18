import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

import scala.jdk.javaapi.CollectionConverters;

import static java.lang.Thread.sleep;

public class ventanaMatriz extends JFrame implements ActionListener {

    JButton boton;
    private JPanel ventana;
    private JTextPane titulo;
    private JRadioButton holaRadioButton;
    private JPanel ventanaDatos;
    private JPanel tablaPanel;

    private int filas = 10;

    private int columnas = 10;

    private JTable tablaCaramelos;

    private DefaultTableModel modeloCaramelos; //Para modificar la tabla

    private int modoJuego;

    private int dificultad = 6;


    scala.collection.immutable.List<Object> matrizScala;

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
        setTitle("Ventana2 de Candy Crush Hola");
        setSize(400, 200);
        setLocationRelativeTo(null); // Centrar la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Crear un panel para colocar los componentes
        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());

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
            tablaCaramelos.getColumnModel().getColumn(i).setCellRenderer(new ColumnColorRenderer());
        }

        tablaCaramelos.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                // do some actions here, for example
                // print first column value from selected row
                int row = 0;
                int col = 0;
                if (modo == 1) {
                    row = tablaCaramelos.rowAtPoint(evt.getPoint());
                    col = tablaCaramelos.columnAtPoint(evt.getPoint());
                } else if (modo == 2) {
                    int pos = Main.getMejorOpcion(matrizScala, filas, columnas, 0, -1, 0);
                    row = pos / columnas;
                    col = pos % columnas;
                }

                System.out.println(row + " " + col);
                scala.collection.immutable.List<Object> result =
                        Main.bucleJuego(matrizScala, columnas, filas, dificultad, 5, 1, row, col);

                if (result.isEmpty()) {
                    System.out.println("Fallo");
                } else {
                    matrizScala = result;
                    matrizScala = Main.flotarCeros(matrizScala, filas, columnas, 0);
                    matrizScala = Main.rellenarTablero(dificultad, matrizScala);
                    actualizarMatriz();
                }
            }
        });
        tablaCaramelos.setVisible(true);
        panel.add(tablaCaramelos);

        // Agregar el panel a la ventana
        this.add(panel);
        this.setVisible(true);

    }

    public void actionPerformed(ActionEvent e) {

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

}