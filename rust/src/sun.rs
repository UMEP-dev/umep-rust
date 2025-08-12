use ndarray::{s, Array2, ArrayView2, Zip};

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
    lc_grid: Option<ArrayView2<f32>>,
    use_landcover: bool,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let (sizex, sizey) = (wall_ht.nrows(), wall_ht.ncols());
    let mut sunwall_mut = sunwall.to_owned();
    sunwall_mut.mapv_inplace(|x| if x > 0. { 1. } else { x });

    let azimuth = azimuth_a * (PI / 180.);

    let mut f = buildings.to_owned();

    let lup = Zip::from(emis_grid)
        .and(tground)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + t_air + 273.15).powi(4) - sbc * emis * (t_air + 273.15).powi(4)
        });

    let mut tground_mut = tground.to_owned();
    if use_landcover {
        if let Some(lc_grid) = lc_grid {
            Zip::from(&mut tground_mut)
                .and(lc_grid)
                .for_each(|tg, &lc| {
                    if lc == 3. {
                        *tg = t_water - t_air;
                    }
                });
        }
    }

    let lwall = sbc * wall_emmisiv * (tg_wall + t_air + 273.15).powi(4)
        - sbc * wall_emmisiv * (t_air + 273.15).powi(4);

    let albshadow = &alb_grid * &shadow;
    let alb = alb_grid;

    let mut tempbu = Array2::<f32>::zeros((sizex, sizey));
    let mut tempsh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempLupsh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempalbsh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempalbnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempwallsun = Array2::<f32>::zeros((sizex, sizey));

    let mut tempbub = Array2::<f32>::zeros((sizex, sizey));
    let mut tempbubwall = Array2::<f32>::zeros((sizex, sizey));

    let mut weightsumsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumLupsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumLwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbsh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwall = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwallnosh = Array2::<f32>::zeros((sizex, sizey));

    let first = (first_ht * pixel_scale).round().max(1.);
    let second = (second_ht * pixel_scale).round();

    let mut weightsumwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumLwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumLupsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwall_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbsh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwallnosh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbnosh_first = Array2::<f32>::zeros((sizex, sizey));

    let pibyfour = PI / 4.;
    let threetimespibyfour = 3. * pibyfour;
    let fivetimespibyfour = 5. * pibyfour;
    let seventimespibyfour = 7. * pibyfour;
    let sinazimuth = azimuth.sin();
    let cosazimuth = azimuth.cos();
    let tanazimuth = azimuth.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();

    for n in 0..second as i32 {
        let index = n as f32;
        let (dx, dy) = if (pibyfour..threetimespibyfour).contains(&azimuth)
            || (fivetimespibyfour..seventimespibyfour).contains(&azimuth)
        {
            (
                -1. * signcosazimuth * (index / tanazimuth).abs().round(),
                signsinazimuth * index,
            )
        } else {
            (
                -1. * signcosazimuth * index,
                signsinazimuth * (index * tanazimuth).abs().round(),
            )
        };

        let dx = dx as isize;
        let dy = dy as isize;

        let absdx = dx.abs();
        let absdy = dy.abs();

        let xc1 = (dx + absdx) / 2;
        let xc2 = sizex as isize + (dx - absdx) / 2;
        let yc1 = (dy + absdy) / 2;
        let yc2 = sizey as isize + (dy - absdy) / 2;

        let xp1 = -(dx - absdx) / 2;
        let xp2 = sizex as isize - (dx + absdx) / 2;
        let yp1 = -(dy - absdy) / 2;
        let yp2 = sizey as isize - (dy + absdy) / 2;

        let x_c_slice = s![xc1..xc2, yc1..yc2];
        let x_p_slice = s![xp1..xp2, yp1..yp2];

        tempbu
            .slice_mut(x_p_slice)
            .assign(&buildings.slice(x_c_slice));
        tempsh.slice_mut(x_p_slice).assign(&shadow.slice(x_c_slice));
        tempLupsh.slice_mut(x_p_slice).assign(&lup.slice(x_c_slice));
        tempalbsh
            .slice_mut(x_p_slice)
            .assign(&albshadow.slice(x_c_slice));
        tempalbnosh
            .slice_mut(x_p_slice)
            .assign(&alb.slice(x_c_slice));

        Zip::from(f.view_mut())
            .and(tempbu.view())
            .for_each(|f_val, &tempbu_val| {
                *f_val = f_val.min(tempbu_val);
            });

        let shadow2 = &tempsh * &f;
        weightsumsh += &shadow2;

        let lupsh = &tempLupsh * &f;
        weightsumLupsh += &lupsh;

        let albsh = &tempalbsh * &f;
        weightsumalbsh += &albsh;

        let albnosh = &tempalbnosh * &f;
        weightsumalbnosh += &albnosh;

        tempwallsun
            .slice_mut(x_p_slice)
            .assign(&sunwall_mut.slice(x_c_slice));
        let tempb = &tempwallsun * &f;
        let tempbwall = &f * -1. + 1.;

        tempbub.zip_mut_with(&tempb, |bub_val, &b| {
            *bub_val = if *bub_val + b > 0. { 1. } else { 0. };
        });
        tempbubwall.zip_mut_with(&tempbwall, |bubwall_val, &bwall| {
            *bubwall_val = if *bubwall_val + bwall > 0. { 1. } else { 0. };
        });

        weightsumLwall.zip_mut_with(&tempbub, |w, &b| *w += b * lwall);
        weightsumalbwall.zip_mut_with(&tempbub, |w, &b| *w += b * wall_albedo);
        weightsumwall.zip_mut_with(&tempbub, |w, &b| *w += b);
        weightsumalbwallnosh.zip_mut_with(&tempbubwall, |w, &b| *w += b * wall_albedo);

        if (n + 1) as f32 <= first {
            // Direct snapshot (no division) per Python sunonsurface_2018a behavior
            weightsumwall_first.assign(&weightsumwall);
            weightsumsh_first.assign(&weightsumsh);
            weightsumLwall_first.assign(&weightsumLwall);
            weightsumLupsh_first.assign(&weightsumLupsh);
            weightsumalbwall_first.assign(&weightsumalbwall);
            weightsumalbsh_first.assign(&weightsumalbsh);
            weightsumalbwallnosh_first.assign(&weightsumalbwallnosh);
            weightsumalbnosh_first.assign(&weightsumalbnosh);
        }
    }

    let wallsuninfluence_first = weightsumwall_first.mapv(|x| (x > 0.) as i32 as f32);
    let wallinfluence_first = weightsumalbwallnosh_first.mapv(|x| (x > 0.) as i32 as f32);
    let wallsuninfluence_second = weightsumwall.mapv(|x| (x > 0.) as i32 as f32);
    let wallinfluence_second = weightsumalbwallnosh.mapv(|x| (x > 0.) as i32 as f32);

    let azilow = azimuth - PI / 2.;
    let azihigh = azimuth + PI / 2.;

    let wallbol = wall_ht.mapv(|x| if x > 0. { 1. } else { 0. });
    let mut facesh;

    if azilow >= 0. && azihigh < 2. * PI {
        facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect < azilow || aspect >= azihigh {
                1.
            } else {
                0.
            }
        });
        facesh = facesh - &wallbol + 1.;
    } else if azilow < 0. && azihigh <= 2. * PI {
        let azilow_adj = azilow + 2. * PI;
        facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect > azilow_adj || aspect <= azihigh {
                -1.
            } else {
                0.
            }
        });
        facesh.mapv_inplace(|x| x + 1.);
    } else {
        // azilow > 0. && azihigh >= 2. * PI
        let azihigh_adj = azihigh - 2. * PI;
        facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect > azilow || aspect <= azihigh_adj {
                -1.
            } else {
                0.
            }
        });
        facesh.mapv_inplace(|x| x + 1.);
    }

    let mut keep = Array2::<f32>::zeros((sizex, sizey));
    Zip::from(&mut keep)
        .and(&weightsumwall)
        .and(&facesh)
        .for_each(|k, &w, &fsh| {
            let val = (if w == second { 1. } else { 0. }) - fsh;
            *k = if val == -1. { 0. } else { val };
        });

    let gvf1 = ((&weightsumwall_first + &weightsumsh_first) / (first + 1.))
        * &wallsuninfluence_first
        + (&weightsumsh_first / first) * (wallsuninfluence_first.mapv(|x| 1. - x));

    let mut weightsumwall_mut = weightsumwall.to_owned();
    weightsumwall_mut.zip_mut_with(&keep, |w, &k| {
        if k == 1. {
            *w = 0.;
        }
    });

    let mut gvf2 = ((&weightsumwall_mut + &weightsumsh) / (second + 1.)) * &wallsuninfluence_second
        + (&weightsumsh / second) * (wallsuninfluence_second.mapv(|x| 1. - x));
    gvf2.mapv_inplace(|x| if x > 1.0 { 1.0 } else { x });

    let gvfLup1 = ((&weightsumLwall_first + &weightsumLupsh_first) / (first + 1.))
        * &wallsuninfluence_first
        + (&weightsumLupsh_first / first) * (wallsuninfluence_first.mapv(|x| 1. - x));

    let mut weightsumLwall_mut = weightsumLwall.to_owned();
    weightsumLwall_mut.zip_mut_with(&keep, |w, &k| {
        if k == 1. {
            *w = 0.;
        }
    });

    let gvfLup2 = ((&weightsumLwall_mut + &weightsumLupsh) / (second + 1.))
        * &wallsuninfluence_second
        + (&weightsumLupsh / second) * (wallsuninfluence_second.mapv(|x| 1. - x));

    let gvfalb1 = ((&weightsumalbwall_first + &weightsumalbsh_first) / (first + 1.))
        * &wallsuninfluence_first
        + (&weightsumalbsh_first / first) * (wallsuninfluence_first.mapv(|x| 1. - x));

    let mut weightsumalbwall_mut = weightsumalbwall.to_owned();
    weightsumalbwall_mut.zip_mut_with(&keep, |w, &k| {
        if k == 1. {
            *w = 0.;
        }
    });

    let gvfalb2 = ((&weightsumalbwall_mut + &weightsumalbsh) / (second + 1.))
        * &wallsuninfluence_second
        + (&weightsumalbsh / second) * (wallsuninfluence_second.mapv(|x| 1. - x));

    let gvfalbnosh1 = ((&weightsumalbwallnosh_first + &weightsumalbnosh_first) / (first + 1.))
        * &wallinfluence_first
        + (&weightsumalbnosh_first / first) * (wallinfluence_first.mapv(|x| 1. - x));
    let gvfalbnosh2 = ((&weightsumalbwallnosh + &weightsumalbnosh) / second)
        * &wallinfluence_second
        + (&weightsumalbnosh / second) * (wallinfluence_second.mapv(|x| 1. - x));

    let gvf = (&gvf1 * 0.5 + &gvf2 * 0.4) / 0.9;

    let buildings_inv = buildings.mapv(|x| 1. - x);

    let lup_final = Zip::from(emis_grid)
        .and(&tground_mut)
        .and(shadow)
        .map_collect(|&emis, &tg, &sh| {
            sbc * emis * (tg * sh + t_air + 273.15).powi(4) - sbc * emis * (t_air + 273.15).powi(4)
        });
    let gvfLup = (&gvfLup1 * 0.5 + &gvfLup2 * 0.4) / 0.9 + &lup_final * &buildings_inv;

    let gvfalb = (&gvfalb1 * 0.5 + &gvfalb2 * 0.4) / 0.9 + &alb_grid * &buildings_inv * &shadow;

    let gvfalbnosh =
        ((&gvfalbnosh1 * 0.5 + &gvfalbnosh2 * 0.4) / 0.9) * &buildings + &alb_grid * &buildings_inv;

    (gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)
}
