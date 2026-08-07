"""Muestra un resumen del dataset YOLO a partir de data.yaml:
numero de imagenes en train/valid y las clases configuradas.
"""

import os
import yaml

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def contar_imagenes(carpeta):
    """Cuenta archivos de imagen en una carpeta (None si la carpeta no existe)."""
    if not os.path.isdir(carpeta):
        return None
    return sum(
        1 for f in os.listdir(carpeta) if f.lower().endswith(IMG_EXTENSIONS)
    )


def resolver_carpeta(base_dir, ruta_relativa):
    """Prueba a resolver la ruta tal cual la indica el yaml y, si no
    existe, tambien relativa directamente a la carpeta de data.yaml
    (algunos exports de Roboflow incluyen un '../' de mas)."""
    candidatas = [
        os.path.normpath(os.path.join(base_dir, ruta_relativa)),
        os.path.normpath(os.path.join(base_dir, ruta_relativa.lstrip("./").lstrip("../"))),
    ]
    for candidata in candidatas:
        if os.path.isdir(candidata):
            return candidata
    return candidatas[0]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "data.yaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_rel = data.get("train")
    val_rel = data.get("val")
    nombres_clases = data.get("names", [])

    train_dir = resolver_carpeta(base_dir, train_rel)
    val_dir = resolver_carpeta(base_dir, val_rel)

    n_train = contar_imagenes(train_dir)
    n_val = contar_imagenes(val_dir)

    print("=== Resumen del dataset ===")
    print(f"Imagenes en train: {n_train if n_train is not None else 'carpeta no encontrada'} ({train_dir})")
    print(f"Imagenes en valid: {n_val if n_val is not None else 'carpeta no encontrada'} ({val_dir})")

    print(f"\nNumero de clases: {len(nombres_clases)}")
    print("Clases de frutas configuradas:")
    for i, nombre in enumerate(nombres_clases):
        print(f"  {i}: {nombre}")


if __name__ == "__main__":
    main()
