"""
Exemple d'ús de Papermill per executar el notebook amb diferents paràmetres.

Aquest script processa múltiples estrelles amb la configuració òptima per a cadascuna.

Instal·lació:
    pip install papermill

Ús:
    python executar_parametritzat.py
"""

import papermill as pm
from configuracio_estrelles import CONFIGURACIONS_ESTRELLES, llista_estrelles
import os

# Directori per guardar els notebooks executats
OUTPUT_NOTEBOOKS = 'output/notebooks'
os.makedirs(OUTPUT_NOTEBOOKS, exist_ok=True)

print("=" * 70)
print("PROCESSAMENT D'ESTRELLES AMB PAPERMILL")
print("=" * 70)

# Processar cada estrella amb la seva configuració específica
for nom_estrella in llista_estrelles():
    config = CONFIGURACIONS_ESTRELLES[nom_estrella]
    
    print(f"\n🌟 Processant: {nom_estrella}")
    print(f"   {config['descripcio']}")
    print(f"   Fitxer: {config['fitxer']}")
    print(f"   Rang de selecció: {config['freq_range_min']}-{config['freq_range_max']} {config['freq_unit']}")
    print(f"   Prominence: {config['prominence']} dB, Distance: {config['distance']} mostres")
    print(f"   Bin width: {config['bin_width']} {config['freq_unit']}")
    
    # Executar el notebook amb els paràmetres específics
    output_notebook = os.path.join(OUTPUT_NOTEBOOKS, f'{nom_estrella}_analisi.ipynb')
    
    # Paràmetres per la funció unificada process_spectrum
    params = {
        'DATA_FILE': config['fitxer'],
        'FREQ_UNIT': config['freq_unit'],
        'FREQ_RANGE_MIN': config['freq_range_min'],
        'FREQ_RANGE_MAX': config['freq_range_max'],
        'PROMINENCE': config['prominence'],
        'DISTANCE': config['distance'],
        'NUM_PEAKS': config['num_peaks'],
        'BIN_WIDTH': config['bin_width'],
        'OUTPUT_DIR': config['output_dir'],
        'CALCULAR_AUTOCORRELACIO': config['calcular_autocorrelacio'],
        'EXCLUDE_NEAR_ZERO': config['exclude_near_zero']
    }
    
    try:
        pm.execute_notebook(
            'analisi.ipynb',
            output_notebook,
            parameters=params
        )
        print(f"   ✅ Completat: {output_notebook}")
        print(f"   📁 Resultats: {config['output_dir']}/")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("PROCESSAMENT COMPLETAT")
print("=" * 70)
print(f"\nNotebooks executats guardats a: {OUTPUT_NOTEBOOKS}/")
print("Resultats CSV guardats a: output/<nom_estrella>/")

