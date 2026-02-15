# Radiation Model

SOLWEIG computes the complete 3D radiation environment for a standing human.

## Radiation Components

### Shortwave (Solar)

| Component | Symbol | Description |
|-----------|--------|-------------|
| Direct | $K_{dir}$ | Direct beam from sun |
| Diffuse | $K_{dif}$ | Scattered by atmosphere |
| Reflected | $K_{ref}$ | Reflected from surfaces |

### Longwave (Thermal)

| Component | Symbol | Description |
|-----------|--------|-------------|
| Downwelling | $L_{down}$ | From sky and atmosphere |
| Upwelling | $L_{up}$ | From ground |
| Lateral | $L_{side}$ | From walls and surfaces |

## Six-Direction Model

Radiation is computed for six directions around a standing person:

- **Up**: Sky/canopy radiation
- **Down**: Ground radiation
- **North, South, East, West**: Lateral radiation from walls

## Direct/Diffuse Split

Global radiation is split into direct and diffuse components using the Reindl model:

$$K_{dir} = K_{global} \times (1 - k_d)$$
$$K_{dif} = K_{global} \times k_d$$

Where $k_d$ is the diffuse fraction, estimated from clearness index.

## Anisotropic vs Isotropic Sky

**Isotropic**: Assumes uniform sky radiance (faster)

**Anisotropic**: Models non-uniform sky brightness (more accurate):

- Circumsolar brightening near sun
- Horizon brightening
- Zenith darkening

## Wall Radiation

Walls contribute lateral radiation based on:

- Wall temperature (function of orientation and solar exposure)
- Wall emissivity
- View factor from point to wall

## Cylindric Weighting

For a standing human (approximated as cylinder), radiation from different directions is weighted:

$$K_{absorbed} = a_k \sum_i w_i K_i$$

Where $w_i$ are direction-dependent weighting factors.

## References

- Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0. *International Journal of Biometeorology*, 52(7), 697-713.
- Lindberg, F., Onomura, S., & Grimmond, C. S. B. (2016). Influence of ground surface characteristics on the mean radiant temperature in urban areas. *International Journal of Biometeorology*, 60(9), 1439-1452.
- Reindl, D. T., Beckman, W. A., & Duffie, J. A. (1990). Diffuse fraction correlations. *Solar Energy*, 45(1), 1-7.
- Perez, R., Seals, R., & Michalsky, J. (1993). All-weather model for sky luminance distribution. *Solar Energy*, 50(3), 235-245.
- Jonsson, P., Eliasson, I., Holmer, B., & Grimmond, C. S. B. (2006). Longwave incoming radiation in the Tropics: results from field work in three African cities. *Theoretical and Applied Climatology*, 85, 185-201.
