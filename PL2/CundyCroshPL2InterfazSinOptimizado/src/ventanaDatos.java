import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ventanaDatos extends JFrame implements ActionListener {

    JButton boton;
    private JPanel ventana;
    private JTextPane titulo;
    private JRadioButton holaRadioButton;
    private JButton pulsaButton;
    private JPanel ventanaDatos;

    public ventanaDatos() {

        // Establecer las propiedades de la ventana
        setTitle("Ventana2 de Candy Crush Hola");
        setSize(400, 200);
        setLocationRelativeTo(null); // Centrar la ventana en la pantalla
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Crear un panel para colocar los componentes
        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());

        // Crear un label con el nombre
        JLabel nombreLabel = new JLabel("Otra vez en Cundy Crush!");
        nombreLabel.setHorizontalAlignment(JLabel.CENTER); // Centrar el texto
        panel.add(nombreLabel, BorderLayout.CENTER);
        panel.add(holaRadioButton);

        // Crear un botón para redirigir a la ventana 2
        boton = new JButton("Ir a ventana 2");
        boton.addActionListener(this); // Agregar un ActionListener
        panel.add(boton, BorderLayout.SOUTH);

        // Agregar el panel a la ventana
        getContentPane().add(panel);

    }

    // Implementar el método actionPerformed para el botón
    public void actionPerformed(ActionEvent e) {

        // Si el botón es pulsado, abrir la ventana 2
        if (e.getSource() == boton) {
            System.out.println("Pulsado");
        }

    }


}
