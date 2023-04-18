
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ventanaDatos extends JFrame implements ActionListener {

    //Elementos de la ventana:
    private JButton botonJugar;
    private JPanel ventanaDatos;
    private JRadioButton modoJuegoBoton1 = new JRadioButton("Manual");
    private JRadioButton modoJuegoBoton2 = new JRadioButton("Automático");
    private JRadioButton dificultadBoton1 = new JRadioButton("Fácil");
    private JRadioButton dificultadBoton2 = new JRadioButton("Difícil");

    //Configuramos los Spinner (Debemos hacer uno por cada spinner porque si no funcionan ambos Spinners como si fuesen solo 1)
    //Establecemos, arbitrariamente, que la matriz más grande posible es 50x50, aunque podríamos aumentar estos valores
    private SpinnerNumberModel formatoFilas = new SpinnerNumberModel(1,1,50,1);
    private SpinnerNumberModel formatoColumnas = new SpinnerNumberModel(1,1,50,1);
    private JSpinner numFilasSpinner = new JSpinner(formatoFilas);
    private JSpinner numColumnasSpinner = new JSpinner(formatoColumnas);

    private int modoJuegoSeleccionado=0;
    private int dificultadSeleccionada=0;
    private int numFilasSeleccionadas=0;
    private int numColumnasSeleccionadas=0;

    public ventanaDatos() {

        // Establecemos las propiedades de la ventana
        setTitle("Selector de opciones - Cundy Crosh");
        setSize(800, 500);
        setLocationRelativeTo(null); // Centramos la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Hacemos que la ventana no sea redimensionable
        setResizable(false);

        // Configuramos el Layout de la ventana
        ventanaDatos.setLayout(null);

        // Creamos el ImageIcon
        ImageIcon imagenFondo = new ImageIcon("src/fondoPantallaCandyCrushOscuro.png");
        // Cargamos la imagen y creamos un objeto JLabel con la imagen
        JLabel labelFondo = new JLabel(imagenFondo);
        // Agregamos el JLabel al panel y establecemos su posición para ponerlo en el fondo
        labelFondo.setBounds(0, 0, imagenFondo.getIconWidth(), imagenFondo.getIconHeight());
        ventanaDatos.add(labelFondo);

        // Creamos un Label para cada componente
        JLabel titulo = new JLabel("Opciones de juego");
        JLabel modoJuegoLabel = new JLabel("Modo de juego:");
        JLabel dificultadLabel = new JLabel("Dificultad:");
        JLabel numFilasLabel = new JLabel("Número de filas:");
        JLabel numColumnasLabel = new JLabel("Número de columnas:");

        //Establecemos las características de los Labels
        titulo.setFont(new Font("Tahoma", Font.BOLD, 30));
        titulo.setForeground(Color.WHITE);
        modoJuegoLabel.setFont(new Font("Tahoma", Font.PLAIN, 20));
        modoJuegoLabel.setForeground(Color.WHITE);
        dificultadLabel.setFont(new Font("Tahoma", Font.PLAIN, 20));
        dificultadLabel.setForeground(Color.WHITE);
        numFilasLabel.setFont(new Font("Tahoma", Font.PLAIN, 20));
        numFilasLabel.setForeground(Color.WHITE);
        numColumnasLabel.setFont(new Font("Tahoma", Font.PLAIN, 20));
        numColumnasLabel.setForeground(Color.WHITE);

        // Obtenemos el tamaño de la ventana y del titulo
        Dimension tamañoVentana = getSize();
        Dimension tamañoTexto = titulo.getPreferredSize();
        // Calculamos la posición del titulo, posicion de la que va a derivar el resto de componentes
        int x = (tamañoVentana.width - tamañoTexto.width) / 2;
        int y = (tamañoVentana.height - tamañoTexto.height) - 450;

        // Establecemos la posición de los textos en el panel
        titulo.setBounds(x, y, tamañoTexto.width+10, tamañoTexto.height);
        modoJuegoLabel.setBounds(x-15, y+70, tamañoTexto.width, tamañoTexto.height);
        dificultadLabel.setBounds(x-15, y+150, tamañoTexto.width, tamañoTexto.height);
        numFilasLabel.setBounds(x-15, y+230, tamañoTexto.width, tamañoTexto.height);
        numColumnasLabel.setBounds(x-15, y+310, tamañoTexto.width, tamañoTexto.height);

        //Ahora agregamos los botones que puede manejar el usuario y los configuramos
        ButtonGroup modoJuegoButtonGroup = new ButtonGroup();
        ButtonGroup dificultadButtonGroup = new ButtonGroup();
        modoJuegoButtonGroup.add(modoJuegoBoton1);
        modoJuegoButtonGroup.add(modoJuegoBoton2);
        dificultadButtonGroup.add(dificultadBoton1);
        dificultadButtonGroup.add(dificultadBoton2);
        //Establecemos las dimensiones y demás valores de los Radio Button
        Dimension tamannoModoJuegoBoton1 = modoJuegoBoton1.getPreferredSize();
        Dimension tamannoModoJuegoBoton2 = modoJuegoBoton2.getPreferredSize();
        Dimension tamannoDificultadBoton1 = dificultadBoton1.getPreferredSize();
        Dimension tamannoDificultadBoton2 = dificultadBoton2.getPreferredSize();
        int tamannoTextoBotones = 17;

        modoJuegoBoton1.setOpaque(false);
        modoJuegoBoton1.setBackground(new Color(0,0,0,0));
        modoJuegoBoton1.setFont(new Font("Tahoma", Font.PLAIN, tamannoTextoBotones));
        modoJuegoBoton1.setForeground(Color.WHITE);
        modoJuegoBoton1.setBounds(x+140, y+70, tamannoModoJuegoBoton1.width+20, tamannoModoJuegoBoton1.height);
        modoJuegoBoton2.setFont(new Font("Tahoma", Font.PLAIN, tamannoTextoBotones));
        modoJuegoBoton2.setForeground(Color.WHITE);
        modoJuegoBoton2.setOpaque(false);
        modoJuegoBoton2.setBackground(new Color(0,0,0,0));
        modoJuegoBoton2.setBounds(x+140, y+90, tamannoModoJuegoBoton2.width+20, tamannoModoJuegoBoton2.height);
        dificultadBoton1.setOpaque(false);
        dificultadBoton1.setBackground(new Color(0,0,0,0));
        dificultadBoton1.setFont(new Font("Tahoma", Font.PLAIN, tamannoTextoBotones));
        dificultadBoton1.setForeground(Color.WHITE);
        dificultadBoton1.setBounds(x+140, y+150, tamannoDificultadBoton1.width+10, tamannoDificultadBoton1.height);
        dificultadBoton2.setFont(new Font("Tahoma", Font.PLAIN, tamannoTextoBotones));
        dificultadBoton2.setForeground(Color.WHITE);
        dificultadBoton2.setOpaque(false);
        dificultadBoton2.setBackground(new Color(0,0,0,0));
        dificultadBoton2.setBounds(x+140, y+170, tamannoDificultadBoton2.width+10, tamannoDificultadBoton2.height);

        //Establecemos las coordenadas de los Spinner
        numFilasSpinner.setLocation(x+140,y+235);
        numFilasSpinner.setSize(40,30);
        numColumnasSpinner.setLocation(x+185,y+315);
        numColumnasSpinner.setSize(40,30);

        // Creamos un botón para redirigir a la siguiente ventana
        botonJugar = new JButton("Jugar!");
        botonJugar.addActionListener(this); // Agregar un ActionListener
        botonJugar.setBounds(x+40, y+370, 200, 50);
        botonJugar.setFont(new Font("Tahoma", Font.BOLD, 30));
        botonJugar.setBackground(new Color(96, 176, 244));
        botonJugar.setForeground(Color.WHITE);

        // Agregamos el titulo y los Labels al panel
        labelFondo.add(titulo);
        labelFondo.add(modoJuegoLabel);
        labelFondo.add(dificultadLabel);
        labelFondo.add(numFilasLabel);
        labelFondo.add(numColumnasLabel);
        //Agregamos los botones al panel
        labelFondo.add(modoJuegoBoton1);
        labelFondo.add(modoJuegoBoton2);
        labelFondo.add(dificultadBoton1);
        labelFondo.add(dificultadBoton2);
        //Agregamos los Spinner al panel
        labelFondo.add(numFilasSpinner);
        labelFondo.add(numColumnasSpinner);
        //Agregamos el botón que nos cambia de ventana
        labelFondo.add(botonJugar);

        // Agregamos el panel a la ventana
        getContentPane().add(ventanaDatos);
    }

    //Establecemos lo que debe ocurrir cuando pulsemos el botón de Jugar
    public void actionPerformed(ActionEvent e) {
        // Si el botón es pulsado, abrimos la siguiente ventana
        if (e.getSource() == botonJugar) {
            //Modo de juego seleccionado por el usuario
            if (modoJuegoBoton1.isSelected()) modoJuegoSeleccionado=1;
            else if (modoJuegoBoton2.isSelected()) modoJuegoSeleccionado=2;
            //Dificultad seleccionada por el usuario
            if (dificultadBoton1.isSelected()) dificultadSeleccionada=1;
            else if (dificultadBoton2.isSelected()) dificultadSeleccionada=2;
            //Número de filas y columnas del tablero seleccionados por el usuario
            numFilasSeleccionadas = (Integer) numFilasSpinner.getValue();
            numColumnasSeleccionadas = (Integer) numColumnasSpinner.getValue();

            //Comprobamos que el usuario haya seleccionado alguna opción, de lo contrario se le muestra un mensaje de error
            if(modoJuegoSeleccionado==0 || dificultadSeleccionada==0)
                JOptionPane.showMessageDialog(null, "Faltan datos por seleccionar ._.", "Error", JOptionPane.ERROR_MESSAGE);
            else System.out.println("HOLA"); //TODO: Descomentar cuando tenga la otra clase --> ventanaJugar ventanaNueva = new ventanaJugar(modoJuegoSeleccionado, dificultadSeleccionada, numFilasSeleccionadas, numColumnasSeleccionadas);

            //Prints para debuggear:
            System.out.println("modo de Juego " + modoJuegoSeleccionado);
            System.out.println("dificultad " + dificultadSeleccionada);
            System.out.println("numFilas " + numFilasSeleccionadas);
            System.out.println("numColumnas " + numColumnasSeleccionadas);
            System.out.println("-------------------------------------");
        }
    }
}
