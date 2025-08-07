use ndarray::{par_azip, s, Array2, ArrayView2};
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn sun_on_surface(
    azimuth_a: f32,
    pixel_scale: f32,
    buildings: ArrayView2<f32>,
    shadow: ArrayView2<f32>,
    sunwall: ArrayView2<f32>,
    first_ht: f32,
    second_ht: f32,
    wall_aspect: ArrayView2<f32>,
    wall_ht: ArrayView2<f32>,
    tground: ArrayView2<f32>,
    tg_wall: f32,
    t_air: f32,
    emis_grid: ArrayView2<f32>,
    wall_emmisiv: f32,
    alb_grid: ArrayView2<f32>,
    sbc: f32,
    wall_albedo: f32,
    t_water: f32,
    lc_grid: ArrayView2<f32>,
    use_landcover: bool,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let (sizex, sizey) = (wall_ht.shape()[0], wall_ht.shape()[1]);

    let mut wallbol = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((w in &mut wallbol, &val in &wall_ht) *w = if val > 0.0 { 1.0 } else { 0.0 });

    let mut sunwall_bin = sunwall.to_owned();
    sunwall_bin
        .par_iter_mut()
        .for_each(|v| *v = if *v > 0.0 { 1.0 } else { 0.0 });

    let azimuth = azimuth_a * (PI / 180.0);
    let mut f = buildings.to_owned();
    let mut tg_mut = tground.to_owned();
    if use_landcover {
        par_azip!((tgval in &mut tg_mut, &lc in &lc_grid) if lc == 3.0 { *tgval = t_water - t_air; });
    }
    let mut lup = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((l in &mut lup, &emis in &emis_grid, &tg_val in &tg_mut, &sh in &shadow)
        *l = sbc * emis * ((tg_val * sh + t_air + 273.15).powi(4))
            - sbc * emis * (t_air + 273.15).powi(4)
    );
    let lwall: f32 = sbc * wall_emmisiv * ((tg_wall + t_air + 273.15).powi(4))
        - sbc * wall_emmisiv * (t_air + 273.15).powi(4);
    let albshadow = &alb_grid.to_owned() * &shadow.to_owned();
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
    let first = (first_ht * pixel_scale).round().max(1.0);
    let second = (second_ht * pixel_scale).round();
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
            dst(&mut tempbu, dst_x, minx, dst_y, miny)
                .assign(&buildings.slice(s![src_x..src_x + minx, src_y..src_y + miny]));
            dst(&mut tempsh, dst_x, minx, dst_y, miny)
                .assign(&shadow.slice(s![src_x..src_x + minx, src_y..src_y + miny]));
            dst(&mut temp_lupsh, dst_x, minx, dst_y, miny)
                .assign(&lup.slice(s![src_x..src_x + minx, src_y..src_y + miny]));
            dst(&mut temp_albsh, dst_x, minx, dst_y, miny)
                .assign(&albshadow.slice(s![src_x..src_x + minx, src_y..src_y + miny]));
            dst(&mut temp_albnosh, dst_x, minx, dst_y, miny)
                .assign(&alb.slice(s![src_x..src_x + minx, src_y..src_y + miny]));

            // f = np.min([f, tempbu], axis=0)
            par_azip!((fval in dst(&mut f, dst_x, minx, dst_y, miny), &tbu in &src(&tempbu, dst_x, minx, dst_y, miny)) {
                *fval = fval.min(tbu);
            });

            // shadow2 = tempsh * f
            let shadow2 =
                &src(&tempsh, dst_x, minx, dst_y, miny) * &src(&f, dst_x, minx, dst_y, miny);
            dst(&mut weightsumsh, dst_x, minx, dst_y, miny).zip_mut_with(&shadow2, |a, &b| *a += b);

            // Lupsh = temp_lupsh * f
            let lupsh =
                &src(&temp_lupsh, dst_x, minx, dst_y, miny) * &src(&f, dst_x, minx, dst_y, miny);
            dst(&mut weightsum_lupsh, dst_x, minx, dst_y, miny)
                .zip_mut_with(&lupsh, |a, &b| *a += b);

            // albsh = temp_albsh * f
            let albsh =
                &src(&temp_albsh, dst_x, minx, dst_y, miny) * &src(&f, dst_x, minx, dst_y, miny);
            dst(&mut weightsum_albsh, dst_x, minx, dst_y, miny)
                .zip_mut_with(&albsh, |a, &b| *a += b);

            // albnosh = temp_albnosh * f
            let albnosh =
                &src(&temp_albnosh, dst_x, minx, dst_y, miny) * &src(&f, dst_x, minx, dst_y, miny);
            dst(&mut weightsum_albnosh, dst_x, minx, dst_y, miny)
                .zip_mut_with(&albnosh, |a, &b| *a += b);

            // tempwallsun = sunwall[xc1:xc2, yc1:yc2]
            dst(&mut tempwallsun, dst_x, minx, dst_y, miny).assign(&src(
                &sunwall_bin,
                src_x,
                minx,
                src_y,
                miny,
            ));
            let tempb =
                &src(&tempwallsun, dst_x, minx, dst_y, miny) * &src(&f, dst_x, minx, dst_y, miny);
            let tempbwall = &src(&f, dst_x, minx, dst_y, miny) * -1.0 + 1.0;

            // tempbub = ((tempb + tempbub) > 0) * 1
            let tempbub_prev = src(&tempbub, dst_x, minx, dst_y, miny).to_owned();
            let mut tempbub_new = Array2::<f32>::zeros((minx, miny));
            par_azip!((tbub in &mut tempbub_new, &tb in &tempb, &tbub_prev in &tempbub_prev) {
                *tbub = if tb + tbub_prev > 0.0 { 1.0 } else { 0.0 };
            });
            dst(&mut tempbub, dst_x, minx, dst_y, miny).assign(&tempbub_new);

            // tempbubwall = ((tempbwall + tempbubwall) > 0) * 1
            let tempbubwall_prev = src(&tempbubwall, dst_x, minx, dst_y, miny).to_owned();
            let mut tempbubwall_new = Array2::<f32>::zeros((minx, miny));
            par_azip!((tbubw in &mut tempbubwall_new, &tbw in &tempbwall, &tbubw_prev in &tempbubwall_prev) {
                *tbubw = if tbw + tbubw_prev > 0.0 { 1.0 } else { 0.0 };
            });
            dst(&mut tempbubwall, dst_x, minx, dst_y, miny).assign(&tempbubwall_new);

            // weightsum_lwall = weightsum_lwall + tempbub * lwall
            let lwallprod = &tempbub_new * lwall;
            dst(&mut weightsum_lwall, dst_x, minx, dst_y, miny)
                .zip_mut_with(&lwallprod, |a, &b| *a += b);

            // weightsum_albwall = weightsum_albwall + tempbub * albedo_b
            let albwallprod = &tempbub_new * wall_albedo;
            dst(&mut weightsum_albwall, dst_x, minx, dst_y, miny)
                .zip_mut_with(&albwallprod, |a, &b| *a += b);

            // weightsumwall = weightsumwall + tempbub
            dst(&mut weightsumwall, dst_x, minx, dst_y, miny)
                .zip_mut_with(&tempbub_new, |a, &b| *a += b);

            // weightsum_albwallnosh = weightsum_albwallnosh + tempbubwall * albedo_b
            let albwallnoshprod = &tempbubwall_new * wall_albedo;
            dst(&mut weightsum_albwallnosh, dst_x, minx, dst_y, miny)
                .zip_mut_with(&albwallnoshprod, |a, &b| *a += b);
        }

        if (n as f32 + 1.0) <= first {
            weightsumwall_first.assign(&(&weightsumwall / ind));
            weightsumsh_first.assign(&(&weightsumsh / ind));
            wallsuninfluence_first.zip_mut_with(&weightsumwall_first, |wuf, &wswf| {
                *wuf = if wswf > 0.0 { 1.0 } else { 0.0 };
            });
            weightsum_lwall_first.assign(&(&weightsum_lwall / ind));
            weightsum_lupsh_first.assign(&(&weightsum_lupsh / ind));
            weightsum_albwall_first.assign(&(&weightsum_albwall / ind));
            weightsum_albsh_first.assign(&(&weightsum_albsh / ind));
            weightsum_albwallnosh_first.assign(&(&weightsum_albwallnosh / ind));
            weightsum_albnosh_first.assign(&(&weightsum_albnosh / ind));
            wallinfluence_first.zip_mut_with(&weightsum_albwallnosh_first, |wif, &wawnf| {
                *wif = if wawnf > 0.0 { 1.0 } else { 0.0 };
            });
            ind += 1.0;
        }
        index += 1.0;
    }
    // --- Post-loop logic: wall self-shadowing, face masks, and final output arrays ---
    let mut facesh = Array2::<f32>::zeros((sizex, sizey));
    let azilow = azimuth - PI / 2.0;
    let mut azihigh = azimuth + PI / 2.0;
    if azilow >= 0.0 && azihigh < 2.0 * PI {
        par_azip!((fsh in &mut facesh, &asp in &wall_aspect, &wbol in &wallbol)
            *fsh = (if asp < azilow || asp >= azihigh { 1.0 } else { 0.0 }) - wbol + 1.0
        );
    } else if azilow < 0.0 && azihigh <= 2.0 * PI {
        let azilow_wrapped = azilow + 2.0 * PI;
        par_azip!((fsh in &mut facesh, &asp in &wall_aspect)
            *fsh = (if asp > azilow_wrapped || asp <= azihigh { -1.0 } else { 0.0 }) + 1.0
        );
    } else if azilow > 0.0 && azihigh >= 2.0 * PI {
        azihigh -= 2.0 * PI;
        par_azip!((fsh in &mut facesh, &asp in &wall_aspect)
            *fsh = (if asp > azilow || asp <= azihigh { -1.0 } else { 0.0 }) + 1.0
        );
    }

    // keep = (weightsumwall == second) - facesh
    let mut keep = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((k in &mut keep, &wsw in &weightsumwall, &fsh in &facesh)
        *k = (if (wsw - second).abs() < 1e-6 { 1.0 } else { 0.0 }) - fsh
    );
    keep.par_iter_mut().for_each(|v| {
        if *v == -1.0 {
            *v = 0.0
        }
    });

    // wallsuninfluence_second = weightsumwall > 0
    let mut wallsuninfluence_second = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((wus in &mut wallsuninfluence_second, &wsw in &weightsumwall) *wus = if wsw > 0.0 { 1.0 } else { 0.0 });
    // wallinfluence_second = weightsum_albwallnosh > 0
    let mut wallinfluence_second = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((wis in &mut wallinfluence_second, &wawn in &weightsum_albwallnosh) *wis = if wawn > 0.0 { 1.0 } else { 0.0 });

    // gvf1 = ((weightsumwall_first + weightsumsh_first) / (first + 1)) * wallsuninfluence_first + (weightsumsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvf1 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvf1, &wswf in &weightsumwall_first, &wshf in &weightsumsh_first, &wuf in &wallsuninfluence_first)
        *g = ((wswf + wshf) / (first + 1.0)) * wuf + (wshf / first) * (-1.0 * wuf + 1.0)
    );

    // weightsumwall[keep == 1] = 0
    par_azip!((wsw in &mut weightsumwall, &k in &keep) if k == 1.0 { *wsw = 0.0 });

    // gvf2 = ((weightsumwall + weightsumsh) / (second + 1)) * wallsuninfluence_second + (weightsumsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvf2 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvf2, &wsw in &weightsumwall, &wsh in &weightsumsh, &wus in &wallsuninfluence_second)
        *g = ((wsw + wsh) / (second + 1.0)) * wus + (wsh / second) * (-1.0 * wus + 1.0)
    );
    gvf2.par_iter_mut().for_each(|v| {
        if *v > 1.0 {
            *v = 1.0
        }
    });

    // gvfLup1 = ((weightsum_lwall_first + weightsum_lupsh_first) / (first + 1)) * wallsuninfluence_first + (weightsum_lupsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvf_lup1 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvf_lup1, &wlwf in &weightsum_lwall_first, &wlsf in &weightsum_lupsh_first, &wuf in &wallsuninfluence_first)
        *g = ((wlwf + wlsf) / (first + 1.0)) * wuf + (wlsf / first) * (-1.0 * wuf + 1.0)
    );
    // weightsum_lwall[keep == 1] = 0
    par_azip!((wlw in &mut weightsum_lwall, &k in &keep) if k == 1.0 { *wlw = 0.0 });
    // gvfLup2 = ((weightsum_lwall + weightsum_lupsh) / (second + 1)) * wallsuninfluence_second + (weightsum_lupsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvf_lup2 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvf_lup2, &wlw in &weightsum_lwall, &wls in &weightsum_lupsh, &wus in &wallsuninfluence_second)
        *g = ((wlw + wls) / (second + 1.0)) * wus + (wls / second) * (-1.0 * wus + 1.0)
    );

    // gvfalb1 = ((weightsum_albwall_first + weightsum_albsh_first) / (first + 1)) * wallsuninfluence_first + (weightsum_albsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    let mut gvfalb1 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvfalb1, &wawf in &weightsum_albwall_first, &wasf in &weightsum_albsh_first, &wuf in &wallsuninfluence_first)
        *g = ((wawf + wasf) / (first + 1.0)) * wuf + (wasf / first) * (-1.0 * wuf + 1.0)
    );
    // weightsum_albwall[keep == 1] = 0
    par_azip!((waw in &mut weightsum_albwall, &k in &keep) if k == 1.0 { *waw = 0.0 });
    // gvfalb2 = ((weightsum_albwall + weightsum_albsh) / (second + 1)) * wallsuninfluence_second + (weightsum_albsh) / (second) * (wallsuninfluence_second * -1 + 1)
    let mut gvfalb2 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvfalb2, &waw in &weightsum_albwall, &was in &weightsum_albsh, &wus in &wallsuninfluence_second)
        *g = ((waw + was) / (second + 1.0)) * wus + (was / second) * (-1.0 * wus + 1.0)
    );

    // gvfalbnosh1
    let mut gvfalbnosh1 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvfalbnosh1, &wawnf in &weightsum_albwallnosh_first, &wanf in &weightsum_albnosh_first, &wif in &wallinfluence_first)
        *g = ((wawnf + wanf) / (first + 1.0)) * wif + (wanf / first) * (-1.0 * wif + 1.0)
    );

    // gvfalbnosh2
    let mut gvfalbnosh2 = Array2::<f32>::zeros((sizex, sizey));
    par_azip!((g in &mut gvfalbnosh2, &wawn in &weightsum_albwallnosh, &wan in &weightsum_albnosh, &wis in &wallinfluence_second)
        *g = ((wawn + wan) / second) * wis + (wan / second) * (-1.0 * wis + 1.0)
    );

    // Weighting
    let gvf = (&gvf1 * 0.5 + &gvf2 * 0.4) / 0.9;

    let mut gvf_lup = (&gvf_lup1 * 0.5 + &gvf_lup2 * 0.4) / 0.9;
    let lup_add = {
        let mut temp = Array2::<f32>::zeros((sizex, sizey));
        par_azip!((l in &mut temp, &emis in &emis_grid, &tg_val in &tg_mut, &sh in &shadow, &bldg in &buildings) {
            let term1 = sbc * emis * (tg_val * sh + t_air + 273.15).powi(4);
            let term2 = sbc * emis * (t_air + 273.15).powi(4);
            *l = (term1 - term2) * (-1.0 * bldg + 1.0);
        });
        temp
    };
    gvf_lup += &lup_add;

    let mut gvfalb = (&gvfalb1 * 0.5 + &gvfalb2 * 0.4) / 0.9;
    let alb_add = &alb_grid.to_owned() * &(&buildings.to_owned() * -1.0 + 1.0) * &shadow.to_owned();
    gvfalb += &alb_add;

    let mut gvfalbnosh = (&gvfalbnosh1 * 0.5 + &gvfalbnosh2 * 0.4) / 0.9;
    let albnosh_add = &gvfalbnosh * &buildings.to_owned()
        + &(&alb_grid.to_owned() * &(&buildings.to_owned() * -1.0 + 1.0));
    gvfalbnosh.assign(&albnosh_add);

    (gvf, gvf_lup, gvfalb, gvfalbnosh, gvf2)
}
