# Astrosismologia (TdR Eduard Carrillo González)

Scripts de Python per analitzar l'espectre freqüencial de senyals d'origen sísmic de diverses estrelles i calcular diferents paràmetres de les mateixes com per exemple el seu radi i la seva massa.

## Estructura del projecte

```
TdR/
├── dades/                          # Fitxers CSV amb espectres de freqüència
│   ├── sol.csv
│   ├── estrellaA.csv
│   ├── estrellaB.csv
│   ├── estrellaC.csv
│   ├── estrellaD.csv
│   └── estrellaE.csv
├── output/                         # Resultats generats automàticament
│   ├── notebooks/                  # Notebooks executats per cada objecte
│   └── [estrella]/                 # CSVs amb resultats per cada estrella
│       ├── histogram_*.csv
│       ├── pairwise_differences.csv
│       └── peaks_around_central.csv
├── astrosismologia_utils.py        # Funcions principals d'anàlisi
├── configuracio_estrelles.py       # Paràmetres per cada objecte
├── executar_parametritzat.py       # Script per processar totes les estrelles
├── analisi.ipynb                   # Notebook plantilla d'anàlisi
└── requeriments.txt                # Dependències del projecte
```

## Dependències

Requereix Python 3.8 o superior i els següents paquets:

- **numpy** >= 1.24 (operacions amb arrays)
- **matplotlib** >= 3.7 (visualització de gràfiques)
- **scipy** >= 1.11 (detecció de pics i anàlisi de senyals)
- **jupyter** >= 1.0 (entorn interactiu per notebooks)
- **papermill** >= 2.4 (execució parametritzada de notebooks)

Estan llistats a `requeriments.txt` a l'arrel del repositori.

## Instal·lació

### 1. Instal·lar Python

Si no tens Python instal·lat, descarrega'l des de:
- **Pàgina oficial**: https://www.python.org/downloads/
- **Recomanat**: Python 3.10 o superior

Durant la instal·lació a Windows, marca l'opció **"Add Python to PATH"**.

### 2. Instal·lació de dependències (Windows / PowerShell)

Crea un entorn virtual i instal·la les dependències:

```powershell
python -m venv .venv
python -m pip install -r requeriments.txt
```

## Ús

### Processar totes les estrelles automàticament

```powershell
python executar_parametritzat.py
```

Aquest script processa el Sol i les 5 estrelles amb els paràmetres òptims definits a `configuracio_estrelles.py`, generant:
- Notebooks executats a `output/notebooks/`
- Resultats CSV a `output/[nom_estrella]/`

### Anàlisi interactiva amb Jupyter

```powershell
python -m jupyter notebook analisi.ipynb
```

Això obre el notebook plantilla on pots ajustar paràmetres manualment.