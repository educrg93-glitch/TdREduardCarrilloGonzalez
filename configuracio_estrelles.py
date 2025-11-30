"""
Configuració específica per a cada estrella.

Aquest fitxer conté els paràmetres òptims per analitzar cada objecte estel·lar,
incloent les unitats de freqüència originals i paràmetres adaptats.
"""

# Configuració per estrella
CONFIGURACIONS_ESTRELLES = {
    'sol': {
        'fitxer': 'dades/sol.csv',
        'freq_unit': 'mHz',
        'freq_range_min': 0.0,
        'freq_range_max': 10.0,
        'prominence': 1.0,
        'distance': 5,
        'num_peaks': 50,
        'bin_width': 0.03,
        'output_dir': 'output/sol',
        'descripcio': 'Sol - Espectre en mHz',
        'calcular_autocorrelacio': False,  # Sol no necessita autocorrelació
        'exclude_near_zero': False  # Sol no exclou freqüències baixes a l'histograma
    },
    'estrellaA': {
        'fitxer': 'dades/estrellaA.csv',
        'freq_unit': 'microHz',
        'freq_range_min': 150.0,
        'freq_range_max': 283.443,
        'prominence': 0.5,
        'distance': 2,
        'num_peaks': 300,
        'bin_width': 0.2,
        'output_dir': 'output/estrellaA',
        'descripcio': 'Estrella A - Espectre en µHz',
        'calcular_autocorrelacio': True,
        'exclude_near_zero': True
    },
    'estrellaB': {
        'fitxer': 'dades/estrellaB.csv',
        'freq_unit': 'microHz',
        'freq_range_min': 120.0,
        'freq_range_max': 240.0,
        'prominence': 0.5,
        'distance': 2,
        'num_peaks': 300,
        'bin_width': 0.2,
        'output_dir': 'output/estrellaB',
        'descripcio': 'Estrella B - Espectre en µHz',
        'calcular_autocorrelacio': True,
        'exclude_near_zero': True
    },
    'estrellaC': {
        'fitxer': 'dades/estrellasC.csv',
        'freq_unit': 'microHz',
        'freq_range_min': 100.0,
        'freq_range_max': 150.0,
        'prominence': 0.5,
        'distance': 2,
        'num_peaks': 300,
        'bin_width': 0.2,
        'output_dir': 'output/estrellaC',
        'descripcio': 'Estrella C - Espectre en µHz',
        'calcular_autocorrelacio': True,
        'exclude_near_zero': True
    },
    'estrellaD': {
        'fitxer': 'dades/estrellaD.csv',
        'freq_unit': 'microHz',
        'freq_range_min': 25.0,
        'freq_range_max': 100.0,
        'prominence': 0.5,
        'distance': 2,
        'num_peaks': 300,
        'bin_width': 0.2,
        'output_dir': 'output/estrellaD',
        'descripcio': 'Estrella D - Espectre en µHz',
        'calcular_autocorrelacio': True,
        'exclude_near_zero': True
    }
}


def get_configuracio(estrella: str) -> dict:
    """Obté la configuració per una estrella específica.
    
    Paràmetres:
        estrella: Nom de l'estrella ('sol', 'estrellaA', 'estrellaB', etc.)
    
    Retorna:
        Diccionari amb la configuració de l'estrella
    """
    if estrella not in CONFIGURACIONS_ESTRELLES:
        disponibles = ', '.join(CONFIGURACIONS_ESTRELLES.keys())
        raise ValueError(f"Estrella '{estrella}' no reconeguda. Disponibles: {disponibles}")
    
    return CONFIGURACIONS_ESTRELLES[estrella].copy()


def llista_estrelles() -> list:
    """Retorna la llista d'estrelles configurades."""
    return list(CONFIGURACIONS_ESTRELLES.keys())
