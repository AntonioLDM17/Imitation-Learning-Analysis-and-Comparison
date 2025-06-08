# Análisis y Comparación de Aprendizaje por Imitación

Este repositorio contiene el código para entrenar y evaluar varias técnicas de aprendizaje por imitación sobre entornos de Gymnasium. Los algoritmos implementados son BC, BCO, GAIL, GAIfO, AIRL y SQIL. Cada carpeta dentro de `Code/` agrupa los scripts para entrenar y evaluar un algoritmo concreto.

## Requisitos

Instala las dependencias con

```bash
pip install -r Code/requirements.txt
```

Es recomendable ejecutar los experimentos en un entorno con GPU debido al coste de entrenamiento.

## Flujo de trabajo

1. **Entrenamiento del experto**  
   Entrena una política experta usando PPO, TRPO o SAC. El modelo se guarda en `data/experts/`.
   ```bash
   python Code/train_expert.py --env halfcheetah --policy sac --timesteps 2000000
   ```
   El script crea la ruta del modelo experto y el directorio de logs tal y como se define en el código【F:Code/train_expert.py†L72-L98】.

2. **Generación de demostraciones**  
   A partir del experto se generan trayectorias que se almacenan bajo `data/demonstrations/<N>` donde `<N>` es el número de episodios.
   ```bash
   python Code/generate_demostrations.py --env halfcheetah --policy sac --timesteps 2000000 --num_episodes 100
   ```
   El nombre y carpeta de las demostraciones se construyen en las siguientes líneas del script【F:Code/generate_demostrations.py†L92-L151】.

3. **Entrenamiento de los algoritmos de imitación**  
   Cada subcarpeta contiene un `train_<alg>.py` que lee las demostraciones correspondientes y guarda el modelo entrenado. Por ejemplo para BC:
   ```bash
   python Code/BC/train_bc.py --env halfcheetah --demo_episodes 100 --timesteps 2000000
   ```
   El script busca las demostraciones y define el directorio de salida según se muestra aquí【F:Code/BC/train_bc.py†L52-L66】.

4. **Evaluación de modelos**  
   Una vez entrenados, todos los modelos pueden evaluarse de manera conjunta con:
   ```bash
   python Code/evaluate_every_model.py --root modelos_finales --episodes 100
   ```
   Este script recorre las carpetas de modelos, carga cada política y genera una tabla con la recompensa media y desviación típica【F:Code/evaluate_every_model.py†L118-L172】.

5. **Generación de gráficas**  
   Los resultados del paso anterior pueden representarse mediante `build_figures.py`:
   ```bash
   python Code/build_figures.py --summary modelos_finales/eval_results_100eps.xlsx
   ```
   El script produce diferentes figuras (error bars, heat‑maps…) y las guarda en la carpeta indicada【F:Code/build_figures.py†L240-L283】.

## Estructura de carpetas

- `Code/AIRL`, `Code/BC`, `Code/BCO`, `Code/GAIL`, `Code/GAIfO`, `Code/SQIL`: scripts de entrenamiento y evaluación para cada algoritmo.
- `Code/data/experts`: modelos entrenados del experto.
- `Code/data/demonstrations/<N>`: demostraciones generadas con `<N>` episodios.
- `Code/figures`: destino de las figuras producidas por `build_figures.py`.
- `Code/config`: archivos YAML con los parámetros utilizados en los experimentos.

## Ejemplo rápido

```bash
# 1. Entrenar experto
python Code/train_expert.py --env halfcheetah --policy sac --timesteps 2000000

# 2. Generar 50 trayectorias
python Code/generate_demostrations.py --env halfcheetah --policy sac --timesteps 2000000 --num_episodes 50

# 3. Entrenar BC con esas trayectorias
python Code/BC/train_bc.py --env halfcheetah --demo_episodes 50 --timesteps 2000000

# 4. Evaluar todos los modelos
python Code/evaluate_every_model.py --root modelos_finales --episodes 100

# 5. Crear gráficas
python Code/build_figures.py --summary modelos_finales/eval_results_100eps.xlsx
```

Este README resume las acciones principales para reproducir el pipeline completo del proyecto.
