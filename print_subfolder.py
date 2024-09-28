import os

# Pfad zu deinem 'images'-Ordner
folder_path = 'images'

# Liste aller Unterordner
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

# Ausgabe der Namen der Unterordner
for subfolder in subfolders:
    print(subfolder)
