# Experimental JAX based autograd approach to orbit fitting
# Result: Not nearly as fast as the physics-based approach
import os
os.environ["JAX_ENABLE_X64"] = "True"
import numba
import jax.numpy as np
import jax
# use JIT to precompile functions
from jax import grad, jit
from jax.experimental import optimizers
import numpy.polynomial.polynomial as poly
import pandas as pd
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
import astropy.units as u
from pprint import pprint
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# convert degree to sexagesimal
def to_seg(ra, dec):
    
    # convert ra, extract each value
    ra_hours = ra // (360/24)
    ra %= 360/24
    ra_minutes = ra // (360/(24 * 60))
    ra %= 360/(24 * 60)
    ra_seconds = ra / (360/(24 * 60 * 60))

    # convert dec, extract each value
    # get the sign of dec so we don't have to deal w/ signage
    sign = math.copysign(1, dec)
    dec = abs(dec)
    dec_deg = dec // 1
    dec %= 1
    dec_minutes = dec // (1 / 60)
    dec %= 1/60
    dec_seconds = dec / (1 / (60 * 60))

    # return the final result as a string
    return f"{int(ra_hours):02d}:{int(ra_minutes):02d}:{ra_seconds:08.5f}", f"{'+' if sign > 0 else '-'}{int(dec_deg):02d}:{int(dec_minutes):02d}:{dec_seconds:08.5f}"


# get a 3D rotation matrix with angle and axis
def get_rotation_mat(angle, axis="x"):

    if axis == "x":
        mat = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], dtype=np.float64)
    elif axis == "y":
        mat = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float64)
    elif axis == "z":
        mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float64)

    return mat

class CONSTS:
    k = 0.0172020989484
    c = 173.144643267 # au / day
    obliquity = np.radians(23.4374)
    eq2ec_mat = get_rotation_mat(-obliquity, axis="x")
    ec2eq_mat = get_rotation_mat(obliquity, axis="x")


# find dE with Newton's method for finding f & g
@jit
def find_dE(r2, r2dot, tau, n, a, x_guess, eps=1e-12, iter=5):
    x = x_guess

    # helpful value to just calculate once, no need to keep calculating every loop
    val = np.dot(r2, r2dot) / (n * a ** 2)
    r2_mag = np.linalg.norm(r2)

    # use fixed interation count so that autograd can work properly
    for i in range(iter):
        f_x = x - (1 - r2_mag / a) * np.sin(x) + val * (1 - np.cos(x)) - n * tau
        f_prime = 1 - (1 - r2_mag / a) * np.cos(x) + val * np.sin(x)
        x -= f_x / f_prime

    return x

@jit
def fg_single(tau, r2, r2dot, flag=0, mu=1):
    r2_mag = np.linalg.norm(r2)

    if flag > 0:

        u = mu / (r2_mag ** 3)
        z = np.dot(r2, r2dot) / (r2_mag ** 2)
        q = np.dot(r2dot, r2dot) / (r2_mag ** 2) - u

        f_series_terms = [
            1,
            0,
            - mu/(2 * (r2_mag ** 3)),
            mu * np.dot(r2, r2dot) / (2 * (r2_mag ** 5)),
            (3 * u * q - 15 * u * z ** 2 + u ** 2) / 24
        ][:flag + 1]

        g_series_terms = [
            0,
            1,
            0,
            -mu / (6 * (r2_mag ** 3)),
            (6 * u * z) / 24
        ][:flag + 1]
        
        powers = tau ** np.arange(flag)
        
        f_series_terms = np.array(f_series_terms)
        g_series_terms = np.array(g_series_terms)

        return np.dot(f_series_terms, powers), np.dot(g_series_terms, powers)
    
    elif flag == 0:
        
        a = 1/(2 / r2_mag - np.dot(r2dot, r2dot) / mu)
        n = np.sqrt(mu / a ** 3)
        e = np.sqrt(1 - np.linalg.norm(np.cross(r2, r2dot)) ** 2 / (mu * a))
        
        val1 = np.dot(r2, r2dot) / (n * a ** 2)
        val2 = n * tau - val1
        sign = val1 * np.cos(val2) + (1 - r2_mag / a) * np.sin(val2)
        guess1 = n * tau + np.copysign(0.85 * e, sign) - val1

        dE1 = find_dE(r2, r2dot, tau, n, a, guess1)
        f1 = 1 - a / r2_mag * (1 - np.cos(dE1))
        g1 = tau + 1 / n * (np.sin(dE1) - dE1)

        return (f1, g1)

    else:

        raise ValueError("Invalid Flag")


def get_orbit(r, rdot, t, t0, mu=1, k=CONSTS.k):

    rmag = np.linalg.norm(r)

    ### SEMIMAJOR AXIS
    a = 1/(2 / rmag - np.dot(rdot, rdot) / mu)

    ### ECCENTRICITY
    h = np.cross(r, rdot)
    hmag = np.linalg.norm(h)
    e = np.sqrt(1 - hmag ** 2 / (mu * a))

    ### INCLINATION
    # h[2] = h_z
    i = np.arccos(h[2] / hmag)

    ### LONG OF ASC NODE
    # h[0] = h_x, h[1] = h_y
    omega = np.arctan2(h[0], -h[1])

    ### ARG OF PERIHELION
    ## first find U
    U = np.arctan2(r[2] / np.sin(i), r[0] * np.cos(omega) + r[1] * np.sin(omega))
    ## then find v
    ecosv = a * (1 - e ** 2) / rmag - 1
    esinv = a * (1 - e ** 2) / hmag * np.dot(r, rdot) / rmag
    v = np.arctan2(esinv, ecosv) % (2 * np.pi)
    ## finally find w
    w = U - v

    ### MEAN ANAMOLY AT EPOCH
    ## first find E
    E = np.arccos((1 - rmag/a) / e)

    if v > np.pi:
        E = 2 * np.pi - E

    ## then find mean anamoly rn
    M = E - e * np.sin(E)
    # finally convert to epoch mean anamoly using mean motion
    # convert n from per gaussian day to per day using k
    n = np.sqrt(mu / a ** 3) * k
    M0 = M + n * (t0 - t)
    
    return dict(
        a=a,
        e=e,
        i=np.degrees(i) % 360,
        omega=np.degrees(omega) % 360,
        w=np.degrees(w) % 360,
        M0=np.degrees(M0) % 360
    )

@jit
# with light correction, num_iter is number times of iteration to acc for light correction
def ephem_r(r, rdot, t_list, t2, R_list, num_iter=3, c=CONSTS.c, k=CONSTS.k):

    t_list_orig = np.array(t_list)

    for _ in range(num_iter):
        fs = []
        gs = []
        taus = []
        for t in t_list:
            tau = k * (t - t2)
            f, g = fg_single(tau, r, rdot)
            fs.append([f])
            gs.append([g])
            taus.append(t)
            
        fs = np.array(fs)
        gs = np.array(gs)

        r_pred = fs * r + gs * rdot
        rho_pred = R_list + r_pred
        rho_pred_mags = np.sum(rho_pred ** 2, axis=1) ** 0.5

        t_list = t_list_orig - rho_pred_mags / c

    normed = rho_pred / np.expand_dims(rho_pred_mags, axis=1)

    ra = np.arctan2(normed[:, 1], normed[:, 0])
    dec = np.arcsin(normed[:, 2])

    return np.degrees(ra) % 360, np.degrees(dec)


obs = []
with open("obs.txt") as f:
    lines = [line for line in f.readlines() if line[0] != "#"]
    
    tokens = lines[0].split()
    epoch = float(tokens[0])
    name = " ".join(tokens[1:])

    for line in lines[1:]:

        tokens = line.split()
        jd = float(tokens[0])
        c = SkyCoord(tokens[1], tokens[2], unit=(u.hourangle, u.deg))
        ra = c.ra.rad
        dec = c.dec.rad
        if len(tokens) == 7:
            sun_vector = np.array([float(tokens[3]), float(tokens[4]), float(tokens[5])], dtype=np.float64)
            ra_err = 0
            dec_err = 0
            obscode="500"
        else:
            ra_err = float(tokens[3]) / 3600
            dec_err = float(tokens[4]) / 3600
            obscode = tokens[5]
            query = Horizons(id="10", location=obscode, epochs=jd, id_type="id")
            sun_vector = query.vectors(refplane="earth", aberrations="apparent")[0]
            sun_vector = np.array([sun_vector["x"], sun_vector["y"], sun_vector["z"]], dtype=np.float64)

        obs.append(dict(
            ra=ra,
            dec=dec,
            ra_err=ra_err,
            dec_err=dec_err,
            jd=jd,
            sun_vector=sun_vector,
            obscode=obscode
        ))

obs = pd.DataFrame(obs)

sun_vecs = np.stack(obs["sun_vector"])
times = np.array(obs["jd"], dtype=np.float64)
ras_obs = np.degrees(np.array(obs["ra"], dtype=np.float64))
decs_obs = np.degrees(np.array(obs["dec"], dtype=np.float64))
ras_obs_err = np.array(obs["ra_err"], dtype=np.float64)
decs_obs_err = np.array(obs["dec_err"], dtype=np.float64)

r = np.array([ 0.23026246, -1.50169513, -0.18074596] , dtype=np.float64)
rdot = np.array([0.66525549, 0.56062365, 0.32885496], dtype=np.float64)
t2 = 2459398.85665902

def rms(r, rdot, times, t2, sun_vecs, ras_obs, decs_obs, debug=False):

    ra, dec = ephem_r(r, rdot, times, t2, sun_vecs)

    delta_ras = ras_obs - ra
    delta_dec = decs_obs - dec
    deltas = np.concatenate((delta_ras, delta_dec))
    rms = np.sqrt(np.sum(deltas ** 2) / (len(deltas) - 6))
    if debug:
        print(deltas)
    
    return rms

grad_rms = grad(rms, argnums=[0, 1])

lr = 0.00000001
rmss = []
for i in tqdm(range(1000)):
    rms_val = rms(r, rdot, times, t2, sun_vecs, ras_obs, decs_obs, debug=False)
    print(rms_val)
    rmss.append(rms_val)
    grad_r, grad_rdot = grad_rms(r, rdot, times, t2, sun_vecs, ras_obs, decs_obs)
    r -= grad_r * lr
    rdot -= grad_rdot * lr

print(rmss)

ra, dec = ephem_r(r, rdot, times, t2, sun_vecs)

ras_fit, decs_fit = ephem_r(r, rdot, times, t2, sun_vecs)
delta_ras = ras_obs - ras_fit
delta_dec = decs_obs - decs_fit

for i in range(len(times)):
    print()
    print(f"OBS{i+1}:  ", *to_seg(ras_obs[i], decs_obs[i]))
    print("PRED: ", *to_seg(ras_fit[i], decs_fit[i]))
    print(f"RESD: {delta_ras[i] * 3600:0.3f} {delta_dec[i] * 3600:0.3f}")

pprint(get_orbit(r, rdot, t2, epoch))