# Protocolo de falsación estricta — H2 (proto-diálogo medible)

> Pre-regístralo *antes* de correr nada. Una hipótesis se falsa con un
> estadístico y un umbral fijados de antemano, no con una inspección visual a
> posteriori. Lo que sigue fija ambos.

## 0. Qué dice H2, operacionalizado sin ambigüedad

Texto del paper (§6.3): *cuando dos sistemas auto-referenciales perturban
mutuamente sus fronteras, la componente sinérgica `Syn` de su información
conjunta sube de forma medible **antes** que la información única `Unique` de
cualquiera de los dos.*

Operacionalización:

- Sea `t_c` el instante en que se activa el acoplamiento.
- Para un par fijo `(i,j)`, fuentes = núcleo de 3 bits de `i` y de `j` en `t`;
  *target* = núcleo conjunto en `t+1`.
- Estimación **sobre el ensemble** de `R` réplicas independientes a tiempo fijo
  `t` (NO promediando a lo largo del tiempo de una sola trayectoria).
- Define el **tiempo de aparición** (onset) de una serie `x(t)` como el primer
  `t > t_c` tal que `x(t) > baseline + k·σ_baseline`, con `baseline` y
  `σ_baseline` medidos en la fase aislada `t < t_c`, y `k` **fijado de antemano**
  (recomendado `k = 3`).
- Estadístico de H2: `Δ = onset(Unique) − onset(Syn)`. H2 predice `Δ > 0`.

## 1. Por qué el pipeline original no puede testar esto (evidencia medida)

| Síntoma medido | Causa raíz |
|---|---|
| CA determinista colapsa a ciclos de **periodo 2–4** en ~25 pasos (N=6,10) | Dinámica sin ruido + acoplamiento *mean-field* |
| Ventana de W=300 → **2 celdas ocupadas** de 4096 en el target conjunto | Se promedia el *tiempo* de una sola trayectoria (= el ciclo) |
| BROJA se cuelga | Programa cónico casi-degenerado (62/64 marginales del target en cero) |
| `dit` no instala | Arrastra `pycddlib` (compila cddlib/GMP) |

Las dos primeras filas hacen que cualquier "Syn precede a Unique" sea un
artefacto del ciclo determinista, no una propiedad informacional. La solución
de las filas 3–4 es de ingeniería; la de las filas 1–2 es **científica** y es la
que decide si H2 es testable.

## 2. Diseño de estimación (corrige el fondo, no solo el solver)

1. **Dinámica estocástica** (`dynamics_stochastic.py`): actualización Boltzmann;
   los créditos gobiernan `beta` de forma continua (sin condicionales `if`). La
   contracción analógico→digital, si ocurre, debe emerger como atractor bajo
   estrés, no por truncamiento exógeno.
2. **Acoplamiento local (anillo)**, no *mean-field*. Con *mean-field* todos los
   pares ven el mismo drive común → la PID por pares está dominada por
   redundancia *por construcción* y no puede testar una sinergia *de par* (H2).
3. **Ensemble a tiempo fijo**: `R ≥ 256` réplicas (sube hasta que el estimador
   se estabilice). Soporte verificado: de **2 → 250–356 celdas** ocupadas.
4. **Solver**: `pid_fast.py` (BROJA_2PID + ECOS) con memoización, guarda de
   degeneración y ratio de adecuación. Corre primero `python pid_fast.py`: si
   XOR no da sinergia≈1 y COPY no da única≈1, el orden (target,s1,s2) está mal.

## 3. Adecuación de muestreo y corrección de sesgo (obligatorio)

La sinergia es la componente **más inflada** por submuestreo. Por cada estimación:

- Exige `adequacy = n_muestras / celdas_ocupadas ≥ 4` (idealmente ≥ 10). Si no,
  **grosea el target** (reduce el target conjunto de 6 bits a ≤3 bits con
  `coarse_grain`) o sube `R`. No confíes en estimaciones marcadas `undersampled`.
- **Resta de sesgo por permutación**: usa `pid_bias_corrected`. Estima el suelo
  de sinergia permutando el target (destruye estructura real, conserva marginales
  y tamaño muestral) y réstalo. Reporta `syn_corrected`, no `syn` crudo.

## 4. Réplicas y test estadístico (sin esto no hay falsación)

- Repite el experimento completo con `S ≥ 30` semillas independientes → obtienes
  una distribución de `Δ`.
- Test **una cola** pre-registrado: signo / bootstrap de la mediana de `Δ`.
  Fija `α = 0.01` de antemano.
- **Decisión** (también pre-registrada):
  - `Δ > 0` significativo → **H2 sobrevive este test** (no "probada": sobrevive).
  - `Δ ≤ 0` significativo, o el onset de Syn no precede en *ningún* régimen →
    **H2 FALSEADA** en esta operacionalización.
  - Sin significancia → **INCONCLUSO** (sube `R`, `S`, o revisa el modelo).

## 5. Controles (cazan el auto-engaño)

- **Placebo desacoplado** (tu idea, correcta): alimenta ruido blanco sin
  acoplar. `Δ` debe ser nulo. Si sale `Δ>0`, tu pipeline fabrica sinergia.
- **Surrogado temporal**: permuta el orden temporal de las réplicas. Debe
  destruir cualquier onset ordenado.
- **Surrogado de marginales**: target permutado (el de §3) como línea base de
  sinergia espuria.

## 6. N\* — hazlo con escalado de tamaño finito, no con "el mayor salto"

El `argmax(diff(synergy))` original es frágil y dio el artefacto `N*=3` (que tú
ya sospechaste venía de reducir a un core de 3 celdas — **no pre-truncar**).

- Define un **parámetro de orden** `φ(N)` (p.ej. fracción de agentes en el
  atractor digital, o la meseta de sinergia corregida).
- Calcula la **susceptibilidad** `χ(N) = Var_réplicas[φ(N)]`. El `N*` crítico es
  el **pico de χ(N)**, no el mayor salto de la media.
- Escanea `N = 2…20` con `S` réplicas cada uno; reporta `χ(N)` con barras de
  error. Un pico nítido y reproducible = transición; sin pico = no hay `N*`.
- Cruza con el cumulante de Binder si quieres rigor de transición de fase.

## 7. Batería de robustez (H2 debe sobrevivir TODAS o reportarse como frágil)

Varía y comprueba que la conclusión no cambia: `W`/`R`, esquema de `beta`,
topología de acoplamiento (anillo vs vecinos-k vs aleatoria dispersa), selección
del core (bits 3:6 vs otros), groseado del target. Si H2 solo aparece en una
combinación, eso es un hallazgo negativo, no positivo.

## 8. Una nota honesta sobre el alcance

- Esto testa **una** operacionalización de H2 (la del paper, §6.3 / Predicción
  P2). Un resultado positivo apoya P2; uno negativo refuta *esta lectura*, no
  toda concepción posible de "proto-diálogo".
- **P1 es la apuesta real del paper** (el umbral `N* ≈ (E_protocol/k_BT)^{1/4}`,
  §2.6, lo único que el propio paper reclama como nuevo). P1 **no** se testa con
  PID: requiere contabilidad energética explícita — medir la tasa de producción
  de entropía analógica vs digital y mostrar el cruce. Si tu objetivo último es
  validar la contribución central, prioriza construir el medidor de
  `dS/dt` real (no el placeholder `N**6` de `information_dynamics.py`) y testar
  el cruce de coste, más que H2.

## Orden de trabajo sugerido en Claude Code

1. `python pid_fast.py` → verifica el solver (gates canónicos).
2. Sustituye `analyze_window` para que llame a `pid_fast.pid` sobre el ensemble.
3. Reemplaza `simulate_network` por `simulate_ensemble` (o injerta tu física en
   el template).
4. Implementa onset + `Δ` + los 3 controles del §5.
5. Loop de `S` semillas → test del §4.
6. Barrido de `N` con susceptibilidad (§6).
7. Batería de robustez (§7).
