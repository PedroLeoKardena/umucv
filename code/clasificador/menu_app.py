import cv2 as cv
import os
import sys
import glob
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import importlib
import inspect

from umucv.stream import autoStream
from umucv.util import read_arguments
from metodos import hands

#Carga dinamica de metodos disponibles
METODOS_DISPONIBLES = {}
_base_dir = os.path.dirname(os.path.abspath(__file__))
_metodos_dir = os.path.join(_base_dir, "metodos")
for _archivo in os.listdir(_metodos_dir):
    if _archivo.endswith(".py") and _archivo not in ["__init__.py", "base.py", "hands.py"]:
        _nombre_modulo = _archivo[:-3]
        try:
            _modulo = importlib.import_module(f"metodos.{_nombre_modulo}")
            for _nombre_clase, _obj in inspect.getmembers(_modulo, inspect.isclass):
                if _obj.__module__ == f"metodos.{_nombre_modulo}":
                    METODOS_DISPONIBLES[_nombre_modulo.upper()] = _obj
                    break
        except Exception as e:
            print(f"[Aviso] No se pudo cargar el método '{_nombre_modulo}': {e}")

def my_arguments(parser):
    parser.add_argument('--models', help='carpeta con el conjunto de imágenes a reconocer (obligatorio)', type=str, required=True)
    parser.add_argument('--method', help=f"nombre del método de comparación (obligatorio). Métodos disponibles actualmente: {METODOS_DISPONIBLES}", type=str, required=True)

def iniciar_camara(clasificador):
    print("\n[INFO] Abriendo cámara... Pulsa 'q' en la ventana para salir al menú.")
    for key, frame in autoStream():
        frame, etiqueta = clasificador.clasificar(frame)
        cv.imshow("Clasificador", frame)

    cv.destroyAllWindows()


class MenuApp:
    def __init__(self, root, args, metodo_elegido):
        self.root = root
        self.args = args
        self.metodo_elegido = metodo_elegido
        
        self.root.title("Menú Clasificador")
        self.root.geometry("450x350")
        
        tk.Label(root, text="Seleccione una opción:", font=("Arial", 14)).pack(pady=15)
        
        tk.Button(root, text=f"1. Reconocimiento ({self.args.method})", command=self.opcion_1, font=("Arial", 11)).pack(fill='x', padx=40, pady=5)
        tk.Button(root, text="2. Reconocedor de gestos", command=self.opcion_2, font=("Arial", 11)).pack(fill='x', padx=40, pady=5)
        tk.Button(root, text="3. Añadir modelos a carpeta generales", command=self.opcion_3, font=("Arial", 11)).pack(fill='x', padx=40, pady=5)
        tk.Button(root, text="4. Añadir nuevo método", command=self.opcion_4, font=("Arial", 11)).pack(fill='x', padx=40, pady=5)
        tk.Button(root, text="5. Descargar basico.txt", command=self.opcion_5, font=("Arial", 11)).pack(fill='x', padx=40, pady=5)
        tk.Button(root, text="Salir", command=self.root.quit, font=("Arial", 11), fg="red").pack(fill='x', padx=40, pady=15)

    def obtener_imagenes_paths(self):
        return glob.glob(os.path.join(self.args.models, '*.png')) + \
               glob.glob(os.path.join(self.args.models, '*.jpg')) + \
               glob.glob(os.path.join(self.args.models, '*.jpeg'))

    def opcion_1(self):
        nombre_carpeta = os.path.basename(os.path.normpath(self.args.models))
        if nombre_carpeta in ["gestos_manos", "gestos_manos_internet"]:
            messagebox.showerror("Error", "No está permitido ejecutar el reconocimiento de objetos generales con los modelos de gestos de manos.")
            return

        if self.metodo_elegido not in METODOS_DISPONIBLES:
            messagebox.showerror("Error", f"El método '{self.args.method}' no existe.")
            return
        
        imagenes_paths = self.obtener_imagenes_paths()
        print("Reconociendo objetos con método introducido: ", self.args.method)
        clasificador = METODOS_DISPONIBLES[self.metodo_elegido]()

        print("Precomputando modelos...")
        for ruta in imagenes_paths:
            nombre = os.path.basename(ruta)
            etiqueta = os.path.splitext(nombre)[0]
            img = cv.imread(ruta)
            if img is not None:
                clasificador.precomputar_modelo(etiqueta, img)

        self.root.withdraw()
        iniciar_camara(clasificador)
        self.root.deiconify()

    def opcion_2(self):
        nombre_carpeta = os.path.basename(os.path.normpath(self.args.models))
        if nombre_carpeta not in ["gestos_manos", "gestos_manos_internet"]:
            messagebox.showerror("Error", "No está permitido ejecutar el reconocimiento de gestos de manos con los modelos de objetos generales.")
            return

        imagenes_paths = self.obtener_imagenes_paths()
        print("Reconociendo gestos de manos")
        clasificador = hands.MetodoManos()

        print("Precomputando modelos...")
        for ruta in imagenes_paths:
            nombre = os.path.basename(ruta)
            etiqueta = os.path.splitext(nombre)[0]
            img = cv.imread(ruta)
            if img is not None:
                clasificador.precomputar_modelo(etiqueta, img)

        self.root.withdraw()
        iniciar_camara(clasificador)
        self.root.deiconify()

    def opcion_3(self):
        archivos = filedialog.askopenfilenames(
            title="Seleccionar modelos",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
        )
        if archivos:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dest_dir = os.path.join(base_dir, "modelos", "generales")
            if not os.path.exists(dest_dir):
                if messagebox.askyesno("Confirmación", "La carpeta 'modelos/generales' no existe. ¿Desea crearla y copiar las imágenes a esta carpeta?"):
                    os.makedirs(dest_dir)
                else:
                    return
            else:
                os.makedirs(dest_dir, exist_ok=True)

            copiados = 0
            for arch in archivos:
                try:
                    shutil.copy(arch, dest_dir)
                    print(f"Copiado {arch} a {dest_dir}")
                    copiados += 1
                except Exception as e:
                    print(f"Error copiando {arch}: {e}")
            if copiados > 0:
                messagebox.showinfo("Éxito", f"Se han guardado {copiados} imágenes en {dest_dir}")

    def opcion_4(self):
        respuesta = messagebox.askyesno("Ayuda", "Recuerda que el nuevo método debe tener el formato correcto para poder ser utilizado por el clasificador. Si quiere comprobar el formato, ejecute la opción 5, donde se descargará un fichero .txt de ejemplo con el formato correcto.\n\n¿Desea continuar?")
        if not respuesta:
            return

        archivo = filedialog.askopenfilename(
            title="Seleccionar nuevo método",
            filetypes=[("Scripts de Python", "*.py")]
        )
        if archivo:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dest_dir = os.path.join(base_dir, "metodos")
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy(archivo, dest_dir)
                print(f"Copiado {archivo} a {dest_dir}")
                messagebox.showinfo("Éxito", f"Fichero {os.path.basename(archivo)} copiado a {dest_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo copiar el archivo: {e}")

    def opcion_5(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_basico = os.path.join(base_dir, "basico.txt")
        if not os.path.exists(ruta_basico):
            messagebox.showerror("Error", f"No se encuentra el archivo {ruta_basico}")
            return
        
        dest = filedialog.asksaveasfilename(
            title="Descargar basico.txt",
            initialfile="basico.txt",
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")]
        )
        if dest:
            try:
                shutil.copy(ruta_basico, dest)
                messagebox.showinfo("Éxito", f"Archivo guardado en:\n{dest}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar: {e}")


if __name__ == "__main__":

    args = read_arguments(my_arguments)
    metodo_elegido = args.method.upper()

    if not os.path.isdir(args.models):
        print(f"Error: La ruta '{args.models}' no existe o no es una carpeta.")
        opcion = input("Deseas crear dicha carpeta [Y/n].")
        if opcion == "Y":
            os.makedirs(args.models)
            print(f"Carpeta '{args.models}' creada.")
        else:
            sys.exit(1)

    imagenes_paths = glob.glob(os.path.join(args.models, '*.png')) + \
                    glob.glob(os.path.join(args.models, '*.jpg')) + \
                    glob.glob(os.path.join(args.models, '*.jpeg'))

    if not imagenes_paths:
        print(f"Aviso: No se han encontrado imágenes (.jpg, .jpeg, .png) en '{args.models}'.")
    else:
        print(f"Se han encontrado {len(imagenes_paths)} imágenes en la carpeta '{args.models}'.")

    # Iniciar la interfaz de Tkinter
    root = tk.Tk()
    app = MenuApp(root, args, metodo_elegido)
    root.mainloop()