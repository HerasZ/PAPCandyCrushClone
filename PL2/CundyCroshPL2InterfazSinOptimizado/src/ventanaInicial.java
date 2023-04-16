import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ventanaInicial extends JFrame implements ActionListener {

    JButton botonAdelante;
    private JPanel ventana;
    private JTextPane titulo;

    public ventanaInicial() {

        // Establecer las propiedades de la ventana
        setTitle("Cundy Crosh PAP-2023");
        setSize(700, 400);
        setLocationRelativeTo(null); // Centrar la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Hacer que la ventana no sea redimensionable
        setResizable(false);

        // Crear un panel para colocar los componentes
        JPanel panel = new JPanel();
        panel.setLayout(null);

// Crear el ImageIcon
        ImageIcon imagenFondo = new ImageIcon("C:\\Users\\UAH\\Desktop\\fondoPantallaCandyCrushOscuro.png");      //TODO: Si funciona, cambiarlo por la ruta relativa del proyecto

// Cargar la imagen y crear un objeto JLabel con la imagen
        JLabel labelFondo = new JLabel(imagenFondo);
// Agregar el JLabel al panel y establecer su posición
        labelFondo.setBounds(0, 0, imagenFondo.getIconWidth(), imagenFondo.getIconHeight());
        panel.add(labelFondo);

        // Crear un label con el nombre
        JLabel titulo = new JLabel("Bienvenido a Cundy Crush!");
        JLabel autores = new JLabel("By: DHZ y ABC");
        titulo.setFont(new Font("Arial", Font.BOLD, 30));
        titulo.setForeground(Color.WHITE);
        autores.setFont(new Font("Tahoma", Font.BOLD, 12));
        autores.setForeground(Color.WHITE);
        // Obtener el tamaño de la ventana y del titulo
        Dimension tamañoVentana = getSize();
        Dimension tamañoTexto = titulo.getPreferredSize();
        // Calcular la posición del titulo
        int x = (tamañoVentana.width - tamañoTexto.width) / 2;
        int y = (tamañoVentana.height - tamañoTexto.height) / 2;
        // Establecer la posición del titulo en el panel
        titulo.setBounds(x, y-90, tamañoTexto.width+20, tamañoTexto.height);
        autores.setBounds(x+145, y-60, tamañoTexto.width+20, tamañoTexto.height);
        // Agregar el titulo al panel
        labelFondo.add(titulo);
        labelFondo.add(autores);

        // Crear un botón para redirigir a la siguiente ventana
        botonAdelante = new JButton("Adelante");
        botonAdelante.addActionListener(this); // Agregar un ActionListener
        botonAdelante.setBounds(x+92, y+40, 200, 50);
        botonAdelante.setFont(new Font("Tahoma", Font.BOLD, 30));
        botonAdelante.setBackground(new Color(96, 176, 244));
        botonAdelante.setForeground(Color.WHITE);
        labelFondo.add(botonAdelante);

        // Agregar el panel a la ventana
        getContentPane().add(panel);
    }

    // Implementar el método actionPerformed para el botón
    public void actionPerformed(ActionEvent e) {
        // Si el botón es pulsado, abrir la siguiente ventana
        if (e.getSource() == botonAdelante) {
            ventanaDatos ventanaNueva = new ventanaDatos();
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
