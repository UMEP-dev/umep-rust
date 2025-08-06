use ndarray::{s, Array2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const PI: f32 = std::f32::consts::PI;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn sun_on_surface(
    py: Python,
    azimuth_a: f32,
    scale: f32,
    buildings: PyReadonlyArray2<f32>,
    shadow: PyReadonlyArray2<f32>,
    sunwall: PyReadonlyArray2<f32>,
    first: f32,
    second: f32,
    aspect: PyReadonlyArray2<f32>,
    walls: PyReadonlyArray2<f32>,
    tg: PyReadonlyArray2<f32>,
    tgwall: PyReadonlyArray2<f32>,
    ta: f32,
    emis_grid: PyReadonlyArray2<f32>,
    ewall: f32,
    alb_grid: PyReadonlyArray2<f32>,
    SBC: f32,
    albedo_b: f32,
    twater: f32,
    lc_grid: PyReadonlyArray2<f32>,
    landcover: i32,
) -> PyResult<(
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
)> {
    let buildings = buildings.as_array().to_owned();
    let shadow = shadow.as_array().to_owned();
    let sunwall = sunwall.as_array().to_owned();
    let aspect = aspect.as_array().to_owned();
    let walls = walls.as_array().to_owned();
    let tg = tg.as_array().to_owned();
    let tgwall = tgwall.as_array().to_owned();
    let emis_grid = emis_grid.as_array().to_owned();
    let alb_grid = alb_grid.as_array().to_owned();
    let lc_grid = lc_grid.as_array().to_owned();
    let (sizex, sizey) = (walls.shape()[0], walls.shape()[1]);

    let mut wallbol = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut wallbol)
        .and(&walls)
        .for_each(|w, &val| *w = if val > 0.0 { 1.0 } else { 0.0 });

    let mut sunwall_bin = sunwall.to_owned();
    sunwall_bin.mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });

    let azimuth = azimuth_a * (PI / 180.0);
    let mut f = buildings.to_owned();
    let mut tg_mut = tg.to_owned();
    if landcover == 1 {
        Zip::from(&mut tg_mut).and(&lc_grid).for_each(|tgval, &lc| {
            if lc == 3.0 {
                *tgval = twater - ta;
            }
        });
    }
    let mut lup = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut lup)
        .and(&emis_grid)
        .and(&tg_mut)
        .and(&shadow)
        .for_each(|l, &emis, &tg_val, &sh| {
            *l = SBC * emis * ((tg_val * sh + ta + 273.15).powi(4))
                - SBC * emis * (ta + 273.15).powi(4);
        });
    let mut lwall = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut lwall).and(&tgwall).for_each(|l, &tgw| {
        *l = SBC * ewall * ((tgw + ta + 273.15).powi(4)) - SBC * ewall * (ta + 273.15).powi(4);
    });
    let albshadow = &alb_grid * &shadow;
    let alb = alb_grid.to_owned();

    let mut tempsh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempbu = Array2::<f32>::zeros((sizex, sizey));
    let mut tempbub = Array2::<f32>::zeros((sizex, sizey));
    let mut tempbubwall = Array2::<f32>::zeros((sizex, sizey));
    let mut tempwallsun = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_lupsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_lwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albwallnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut temp_lupsh = Array2::<f32>::zeros((sizex, sizey));
    let mut temp_albsh = Array2::<f32>::zeros((sizex, sizey));
    let mut temp_albnosh = Array2::<f32>::zeros((sizex, sizey));

    let pibyfour = PI / 4.0;
    let threetimespibyfour = 3.0 * pibyfour;
    let fivetimespibyfour = 5.0 * pibyfour;
    let seventimespibyfour = 7.0 * pibyfour;
    let sinazimuth = azimuth.sin();
    let cosazimuth = azimuth.cos();
    let tanazimuth = azimuth.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();

    let mut index = 0.0;
    let first = (first * scale).round().max(1.0);
    let second = (second * scale).round();
    let mut weightsumwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut wallsuninfluence_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_lwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_lupsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albwallnosh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsum_albnosh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut wallinfluence_first = Array2::<f32>::zeros((sizex, sizey));
    let mut ind = 1.0;

    for n in 0..(second as usize) {
        let (dx, dy) = if (pibyfour <= azimuth && azimuth < threetimespibyfour)
            || (fivetimespibyfour <= azimuth && azimuth < seventimespibyfour)
        {
            (
                -1.0 * signcosazimuth * (index / tanazimuth).round().abs(),
                signsinazimuth * index,
            )
        } else {
            (
                -1.0 * signcosazimuth * index,
                signsinazimuth * (index * tanazimuth).round().abs(),
            )
        };
        let absdx = dx.abs();
        let absdy = dy.abs();
        let xc1 = ((dx + absdx) / 2.0) as isize;
        let xc2 = (sizex as f32 + (dx - absdx) / 2.0) as isize;
        let yc1 = ((dy + absdy) / 2.0) as isize;
        let yc2 = (sizey as f32 + (dy - absdy) / 2.0) as isize;
        let xp1 = -((dx - absdx) / 2.0) as isize;
        let xp2 = (sizex as f32 - (dx + absdx) / 2.0) as isize;
        let yp1 = -((dy - absdy) / 2.0) as isize;
        let yp2 = (sizey as f32 - (dy + absdy) / 2.0) as isize;

        // Safe slicing for all temp arrays as in Python
        // Clamp indices to valid ranges
        let xc1c = xc1.max(0).min(sizex as isize);
        let xc2c = xc2.max(0).min(sizex as isize);
        let yc1c = yc1.max(0).min(sizey as isize);
        let yc2c = yc2.max(0).min(sizey as isize);
        let xp1c = xp1.max(0).min(sizex as isize);
        let xp2c = xp2.max(0).min(sizex as isize);
        let yp1c = yp1.max(0).min(sizey as isize);
        let yp2c = yp2.max(0).min(sizey as isize);

        // Only proceed if the slices are valid
        if xc2c > xc1c && yc2c > yc1c && xp2c > xp1c && yp2c > yp1c {
            let xlen = (xc2c - xc1c) as usize;
            let ylen = (yc2c - yc1c) as usize;
            let xplen = (xp2c - xp1c) as usize;
            let yplen = (yp2c - yp1c) as usize;
            let minx = xlen.min(xplen);
            let miny = ylen.min(yplen);
            let src_x = xc1c as usize;
            let src_y = yc1c as usize;
            let dst_x = xp1c as usize;
            let dst_y = yp1c as usize;

            // Slices for source and destination
            fn src<'a, S: ndarray::Data<Elem = f32>>(
                arr: &'a ndarray::ArrayBase<S, ndarray::Ix2>,
                src_x: usize,
                minx: usize,
                src_y: usize,
                miny: usize,
            ) -> ndarray::ArrayView2<'a, f32> {
                arr.slice(s![src_x..src_x + minx, src_y..src_y + miny])
            }
            fn dst<'a>(
                arr: &'a mut Array2<f32>,
                dst_x: usize,
                minx: usize,
                dst_y: usize,
                miny: usize,
            ) -> ndarray::ArrayViewMut2<'a, f32> {
                arr.slice_mut(s![dst_x..dst_x + minx, dst_y..dst_y + miny])
            }

            // tempbu = buildings[xc1:xc2, yc1:yc2]
            // Ensure the slices have the same shape and compatible types
            let src_buildings = src(&buildings, src_x, minx, src_y, miny).to_owned();
            let src_shadow = src(&shadow, src_x, minx, src_y, miny).to_owned();
            dst(&mut tempbu, dst_x, minx, dst_y, miny).assign(&src_buildings);
            dst(&mut tempsh, dst_x, minx, dst_y, miny).assign(&src_shadow);
            dst(&mut temp_lupsh, dst_x, minx, dst_y, miny)
                .assign(&src(&lup, src_x, minx, src_y, miny));
            dst(&mut temp_albsh, dst_x, minx, dst_y, miny)
                .assign(&src(&albshadow, src_x, minx, src_y, miny));
            dst(&mut temp_albnosh, dst_x, minx, dst_y, miny)
                .assign(&src(&alb, src_x, minx, src_y, miny));

            // f = np.min([f, tempbu], axis=0)
            Zip::from(dst(&mut f, dst_x, minx, dst_y, miny))
                .and(dst(&mut tempbu, dst_x, minx, dst_y, miny))
                .for_each(|fval, tbu| {
                    *fval = fval.min(*tbu);
                });

            // shadow2 = tempsh * f
            let tempsh_slice = dst(&mut tempsh, dst_x, minx, dst_y, miny).to_owned();
            let f_slice = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let shadow2 = &tempsh_slice * &f_slice;
            let weightsumsh_slice = dst(&mut weightsumsh, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsumsh, dst_x, minx, dst_y, miny).assign(&(weightsumsh_slice + &shadow2));

            // Lupsh = temp_lupsh * f
            let temp_lupsh_slice = dst(&mut temp_lupsh, dst_x, minx, dst_y, miny).to_owned();
            let f_slice2 = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let lupsh = &temp_lupsh_slice * &f_slice2;
            let weightsum_lupsh_slice =
                dst(&mut weightsum_lupsh, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_lupsh, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_lupsh_slice + &lupsh));

            // albsh = temp_albsh * f
            let temp_albsh_slice = dst(&mut temp_albsh, dst_x, minx, dst_y, miny).to_owned();
            let f_slice3 = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let albsh = &temp_albsh_slice * &f_slice3;
            let weightsum_albsh_slice =
                dst(&mut weightsum_albsh, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_albsh, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_albsh_slice + &albsh));

            // albnosh = temp_albnosh * f
            let temp_albnosh_slice = dst(&mut temp_albnosh, dst_x, minx, dst_y, miny).to_owned();
            let f_slice4 = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let albnosh = &temp_albnosh_slice * &f_slice4;
            let weightsum_albnosh_slice =
                dst(&mut weightsum_albnosh, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_albnosh, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_albnosh_slice + &albnosh));

            // tempwallsun = sunwall[xc1:xc2, yc1:yc2]
            dst(&mut tempwallsun, dst_x, minx, dst_y, miny).assign(&src(
                &sunwall_bin,
                src_x,
                minx,
                src_y,
                miny,
            ));
            let tempwallsun_slice = dst(&mut tempwallsun, dst_x, minx, dst_y, miny).to_owned();
            let f_slice5 = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let tempb = &tempwallsun_slice * &f_slice5;
            let f_slice6 = dst(&mut f, dst_x, minx, dst_y, miny).to_owned();
            let tempbwall = &f_slice6 * -1.0 + 1.0;
            // tempbub = ((tempb + tempbub) > 0) * 1
            let tempbub_prev = dst(&mut tempbub, dst_x, minx, dst_y, miny).to_owned();
            let mut tempbub_new = tempbub_prev.clone();
            Zip::from(&mut tempbub_new)
                .and(&tempb)
                .and(&tempbub_prev)
                .for_each(|tbub, &tb, &tbub_prev| {
                    *tbub = if tb + tbub_prev > 0.0 { 1.0 } else { 0.0 };
                });
            dst(&mut tempbub, dst_x, minx, dst_y, miny).assign(&tempbub_new);
            // tempbubwall = ((tempbwall + tempbubwall) > 0) * 1
            let tempbubwall_prev = dst(&mut tempbubwall, dst_x, minx, dst_y, miny).to_owned();
            let mut tempbubwall_new = tempbubwall_prev.clone();
            Zip::from(&mut tempbubwall_new)
                .and(&tempbwall)
                .and(&tempbubwall_prev)
                .for_each(|tbubw, &tbw, &tbubw_prev| {
                    *tbubw = if tbw + tbubw_prev > 0.0 { 1.0 } else { 0.0 };
                });
            dst(&mut tempbubwall, dst_x, minx, dst_y, miny).assign(&tempbubwall_new);
            // weightsum_lwall = weightsum_lwall + tempbub * lwall
            let lwall_slice = src(&lwall, src_x, minx, src_y, miny).to_owned();
            let tempbub_slice = tempbub_new;
            let lwallprod = &tempbub_slice * &lwall_slice;
            let weightsum_lwall_slice =
                dst(&mut weightsum_lwall, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_lwall, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_lwall_slice + &lwallprod));
            // weightsum_albwall = weightsum_albwall + tempbub * albedo_b
            let albwallprod = &tempbub_slice * albedo_b;
            let weightsum_albwall_slice =
                dst(&mut weightsum_albwall, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_albwall, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_albwall_slice + &albwallprod));
            // weightsumwall = weightsumwall + tempbub
            let weightsumwall_slice = dst(&mut weightsumwall, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsumwall, dst_x, minx, dst_y, miny)
                .assign(&(weightsumwall_slice + &tempbub_slice));
            // weightsum_albwallnosh = weightsum_albwallnosh + tempbubwall * albedo_b
            let tempbubwall_slice = tempbubwall_new;
            let albwallnoshprod = &tempbubwall_slice * albedo_b;
            let weightsum_albwallnosh_slice =
                dst(&mut weightsum_albwallnosh, dst_x, minx, dst_y, miny).to_owned();
            dst(&mut weightsum_albwallnosh, dst_x, minx, dst_y, miny)
                .assign(&(weightsum_albwallnosh_slice + &albwallnoshprod));
        }
        index += 1.0;
    }
    // --- Post-loop logic: wall self-shadowing, face masks, and final output arrays ---
    let mut facesh = Array2::<f32>::zeros((sizex, sizey));
    let azilow = azimuth - PI / 2.0;
    let mut azihigh = azimuth + PI / 2.0;
    if azilow >= 0.0 && azihigh < 2.0 * PI {
        Zip::from(&mut facesh)
            .and(&aspect)
            .and(&wallbol)
            .for_each(|fsh, &asp, &wbol| {
                *fsh = (if asp < azilow || asp >= azihigh {
                    1.0
                } else {
                    0.0
                }) - wbol
                    + 1.0;
            });
    } else if azilow < 0.0 && azihigh <= 2.0 * PI {
        let azilow_wrapped = azilow + 2.0 * PI;
        Zip::from(&mut facesh).and(&aspect).for_each(|fsh, &asp| {
            *fsh = (if asp > azilow_wrapped || asp <= azihigh {
                -1.0
            } else {
                0.0
            }) + 1.0;
        });
    } else if azilow > 0.0 && azihigh >= 2.0 * PI {
        azihigh = azihigh - 2.0 * PI;
        Zip::from(&mut facesh).and(&aspect).for_each(|fsh, &asp| {
            *fsh = (if asp > azilow || asp <= azihigh {
                -1.0
            } else {
                0.0
            }) + 1.0;
        });
    }

    // keep = (weightsumwall == second) - facesh
    let mut keep = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut keep)
        .and(&weightsumwall)
        .and(&facesh)
        .for_each(|k, &wsw, &fsh| {
            *k = if (wsw - second).abs() < 1e-6 {
                1.0
            } else {
                0.0
            } - fsh;
        });
    keep.mapv_inplace(|v| if v == -1.0 { 0.0 } else { v });

    // wallsuninfluence_second = weightsumwall > 0
    let mut wallsuninfluence_second = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut wallsuninfluence_second)
        .and(&weightsumwall)
        .for_each(|wus, &wsw| *wus = if wsw > 0.0 { 1.0 } else { 0.0 });
    // wallinfluence_second = weightsum_albwallnosh > 0
    let mut wallinfluence_second = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut wallinfluence_second)
        .and(&weightsum_albwallnosh)
        .for_each(|wis, &wawn| *wis = if wawn > 0.0 { 1.0 } else { 0.0 });

    // gvf1 = ((weightsumwall_first + weightsumsh_first) / (first + 1)) * wallsuninfluence_first + (weightsumsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvf1 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvf1)
        .and(&weightsumwall_first)
        .and(&weightsumsh_first)
        .and(&wallsuninfluence_first)
        .for_each(|g, &wswf, &wshf, &wuf| {
            *g = ((wswf + wshf) / (first + 1.0)) * wuf + (wshf / first) * (-1.0 * wuf + 1.0);
        });

    // weightsumwall[keep == 1] = 0
    Zip::from(&mut weightsumwall)
        .and(&keep)
        .for_each(|wsw, &k| {
            if k == 1.0 {
                *wsw = 0.0
            }
        });

    // gvf2 = ((weightsumwall + weightsumsh) / (second + 1)) * wallsuninfluence_second + (weightsumsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvf2 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvf2)
        .and(&weightsumwall)
        .and(&weightsumsh)
        .and(&wallsuninfluence_second)
        .for_each(|g, &wsw, &wsh, &wus| {
            *g = ((wsw + wsh) / (second + 1.0)) * wus + (wsh / second) * (-1.0 * wus + 1.0);
        });
    gvf2.mapv_inplace(|v| if v > 1.0 { 1.0 } else { v });

    // gvfLup1 = ((weightsum_lwall_first + weightsum_lupsh_first) / (first + 1)) * wallsuninfluence_first + (weightsum_lupsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvf_lup1 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvf_lup1)
        .and(&weightsum_lwall_first)
        .and(&weightsum_lupsh_first)
        .and(&wallsuninfluence_first)
        .for_each(|g, &wlwf, &wlsf, &wuf| {
            *g = ((wlwf + wlsf) / (first + 1.0)) * wuf + (wlsf / first) * (-1.0 * wuf + 1.0);
        });
    // weightsum_lwall[keep == 1] = 0
    Zip::from(&mut weightsum_lwall)
        .and(&keep)
        .for_each(|wlw, &k| {
            if k == 1.0 {
                *wlw = 0.0
            }
        });
    // gvfLup2 = ((weightsum_lwall + weightsum_lupsh) / (second + 1)) * wallsuninfluence_second + (weightsum_lupsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvf_lup2 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvf_lup2)
        .and(&weightsum_lwall)
        .and(&weightsum_lupsh)
        .and(&wallsuninfluence_second)
        .for_each(|g, &wlw, &wls, &wus| {
            *g = ((wlw + wls) / (second + 1.0)) * wus + (wls / second) * (-1.0 * wus + 1.0);
        });

    // gvfalb1 = ((weightsum_albwall_first + weightsum_albsh_first) / (first + 1)) * wallsuninfluence_first + (weightsum_albsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvfalb1 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvfalb1)
        .and(&weightsum_albwall_first)
        .and(&weightsum_albsh_first)
        .and(&wallsuninfluence_first)
        .for_each(|g, &wawf, &wasf, &wuf| {
            *g = ((wawf + wasf) / (first + 1.0)) * wuf + (wasf / first) * (-1.0 * wuf + 1.0);
        });
    // weightsum_albwall[keep == 1] = 0
    Zip::from(&mut weightsum_albwall)
        .and(&keep)
        .for_each(|waw, &k| {
            if k == 1.0 {
                *waw = 0.0
            }
        });
    // gvfalb2 = ((weightsum_albwall + weightsum_albsh) / (second + 1)) * wallsuninfluence_second + (weightsum_albsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvfalb2 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvfalb2)
        .and(&weightsum_albwall)
        .and(&weightsum_albsh)
        .and(&wallsuninfluence_second)
        .for_each(|g, &waw, &was, &wus| {
            *g = ((waw + was) / (second + 1.0)) * wus + (was / second) * (-1.0 * wus + 1.0);
        });

    // gvfalbnosh1 = ((weightsum_albwallnosh_first + weightsum_albnosh_first) / (first + 1)) * wallinfluence_first + (weightsum_albnosh_first) / (first) * (wallinfluence_first * -1 + 1)
    let mut gvfalbnosh1 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvfalbnosh1)
        .and(&weightsum_albwallnosh_first)
        .and(&weightsum_albnosh_first)
        .and(&wallinfluence_first)
        .for_each(|g, &wawnf, &wanf, &wif| {
            *g = ((wawnf + wanf) / (first + 1.0)) * wif + (wanf / first) * (-1.0 * wif + 1.0);
        });
    // gvfalbnosh2 = ((weightsum_albwallnosh + weightsum_albnosh) / (second)) * wallinfluence_second + (weightsum_albnosh) / (second) * (wallinfluence_second * -1 + 1)
    let mut gvfalbnosh2 = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut gvfalbnosh2)
        .and(&weightsum_albwallnosh)
        .and(&weightsum_albnosh)
        .and(&wallinfluence_second)
        .for_each(|g, &wawn, &wan, &wis| {
            *g = ((wawn + wan) / (second)) * wis + (wan / second) * (-1.0 * wis + 1.0);
        });

    // Weighting
    let mut gvf = (&gvf1 * 0.5 + &gvf2 * 0.4) / 0.9;
    let mut gvf_lup = (&gvf_lup1 * 0.5 + &gvf_lup2 * 0.4) / 0.9;
    gvf_lup = &gvf_lup
        + &((&emis_grid * &((&tg * &shadow + ta + 273.15).mapv(|v| v.powi(4)))
            - &emis_grid * (ta + 273.15).powi(4))
            * (&buildings * -1.0 + 1.0))
            * SBC;
    let mut gvfalb = (&gvfalb1 * 0.5 + &gvfalb2 * 0.4) / 0.9;
    gvfalb = &gvfalb + &(&alb_grid * (&buildings * -1.0 + 1.0) * &shadow);
    let mut gvfalbnosh = (&gvfalbnosh1 * 0.5 + &gvfalbnosh2 * 0.4) / 0.9;
    gvfalbnosh = &gvfalbnosh * &buildings + &alb_grid * (&buildings * -1.0 + 1.0);

    Ok((
        gvf.into_pyarray(py).to_owned().into(),
        gvf_lup.into_pyarray(py).to_owned().into(),
        gvfalb.into_pyarray(py).to_owned().into(),
        gvfalbnosh.into_pyarray(py).to_owned().into(),
        gvf2.into_pyarray(py).to_owned().into(),
    ))
}
