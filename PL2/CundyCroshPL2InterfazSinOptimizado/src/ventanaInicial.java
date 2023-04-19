import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ventanaInicial extends JFrame implements ActionListener {

    private Image iconoVentana = new ImageIcon("src/Imagenes/appIconCandyCrush.png").getImage();
    JButton botonAdelante;
    private JPanel ventana;
    private JTextPane titulo;

    public ventanaInicial() {

        // Establecemos las propiedades de la ventana
        setTitle("Ventana Inicial - Cundy Crosh");
        setSize(700, 400);
        setLocationRelativeTo(null); // Centramos la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Cambiamos el icono de la ventana
        setIconImage(iconoVentana);

        // Hacemos que la ventana no sea redimensionable
        setResizable(false);

        // Creamos un panel para colocar los componentes
        JPanel panel = new JPanel();
        panel.setLayout(null);

        // Creamos el ImageIcon
        ImageIcon imagenFondo = new ImageIcon("src/Imagenes/fondoPantallaCandyCrushOscuro.png");

        // Cargamos la imagen y creamos un objeto JLabel con la imagen
        JLabel labelFondo = new JLabel(imagenFondo);
        // Agregamos el JLabel al panel y establecemos su posición
        labelFondo.setBounds(0, 0, imagenFondo.getIconWidth(), imagenFondo.getIconHeight());
        panel.add(labelFondo);

        // Creamos un label para cada componente
        JLabel titulo = new JLabel("¡Bienvenido a Cundy Crosh!");
        JLabel autores = new JLabel("By: DHZ y ABC");
        titulo.setFont(new Font("Tahoma", Font.BOLD, 30));
        titulo.setForeground(Color.WHITE);
        autores.setFont(new Font("Tahoma", Font.BOLD, 12));
        autores.setForeground(Color.WHITE);
        // Obtenemos el tamaño de la ventana y del titulo
        Dimension tamañoVentana = getSize();
        Dimension tamañoTexto = titulo.getPreferredSize();
        // Calculamos la posición del titulo
        int x = (tamañoVentana.width - tamañoTexto.width) / 2;
        int y = (tamañoVentana.height - tamañoTexto.height) / 2;
        // Establecemos la posición de los textos en el panel
        titulo.setBounds(x, y-90, tamañoTexto.width+20, tamañoTexto.height);
        autores.setBounds(x+145, y-60, tamañoTexto.width+20, tamañoTexto.height);
        // Agregamos el titulo y el subtitulo al panel
        labelFondo.add(titulo);
        labelFondo.add(autores);

        // Creamos un botón para redirigir a la siguiente ventana
        botonAdelante = new JButton("Adelante");
        botonAdelante.addActionListener(this); // Agregar un ActionListener
        botonAdelante.setBounds(x+92, y+40, 200, 50);
        botonAdelante.setFont(new Font("Tahoma", Font.BOLD, 30));
        botonAdelante.setBackground(new Color(96, 176, 244));
        botonAdelante.setForeground(Color.WHITE);
        labelFondo.add(botonAdelante);

        // Agregamos el panel a la ventana
        getContentPane().add(panel);
    }

    //Establecemos lo que debe ocurrir cuando pulsemos el boton
    public void actionPerformed(ActionEvent e) {
        // Si el botón es pulsado, abrimos la siguiente ventana
        if (e.getSource() == botonAdelante) {
            ventanaDatos ventanaNueva = new ventanaDatos();
//            ventanaNueva.setContentPane(new ventanaDatos().ventanaDatos);
            ventanaNueva.setVisible(true);
            setVisible(false);
        }
    }

    // Clase principal para ejecutar la aplicación
    public static void main(String[] args) {
        ventanaInicial ventana1 = new ventanaInicial();
        ventana1.setVisible(true);
    }

}
