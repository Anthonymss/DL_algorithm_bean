import os
import shutil

cache_dir = os.path.expanduser('~/.keras/models')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
print("✅ Caché de modelos Keras borrada.")
