import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
import astropy.units as u
from pprint import pprint
import math

# https://stackoverflow.com/a/23689767
# helpful class to access a dictionary using dots (useful for reprsenting orbits)
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# get a 3D rotation matrix with angle and axis
def get_rotation_mat(angle, axis="x"):
    # create a 2d rotation matrix
    mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])#, dtype=np.float128)

    # convert to a 3d rotation matrix by inserting the 0's and 1's
    num = ["x", "y", "z"].index(axis)
    mat = np.insert(mat, num, [0, 0], axis=1)
    mat = np.insert(mat, num, [0, 0, 0], axis=0)
    mat[num, num] = 1

    return mat

# constants class
class CONSTS:
    k = 0.0172020989484
    c = 173.144643267 # au / day
    obliquity = np.radians(23.4374)
    # generate conversions from equatorial to ecliptic & vice-versa
    eq2ec_mat = get_rotation_mat(-obliquity, axis="x") 
    ec2eq_mat = get_rotation_mat(obliquity, axis="x")

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

# extract rho_hat using formula
def get_rho_hat(ra, dec):
    return np.array(
        [
            np.cos(ra) * np.cos(dec), 
            np.sin(ra) * np.cos(dec),
            np.sin(dec)
        ]
    )

# taus = [tau1, tau3, tau3 - tau1]
# Ds = [D0, D21, D22, D23]
# scalar equation of lagrange to get inital estimates of the magnitude of r2 (roots) and the length of the rho vectors (rhos)
def SEL(taus, Sun2, rhohat2, Ds, mu=1):

    tau1, tau3, tau = taus

    # find expressions for variables
    A1 = tau3 / tau
    B1 = A1 / 6 * (tau ** 2 - tau3 ** 2)
    A3 = -tau1 / tau
    B3 = A3 / 6 * (tau ** 2 - tau1 ** 2)

    A = (A1 * Ds[1] - Ds[2] + A3 * Ds[3]) / (-Ds[0])
    B = (B1 * Ds[1] + B3 * Ds[3]) / (-Ds[0])

    E = -2 * np.dot(rhohat2, Sun2)
    F = np.dot(Sun2, Sun2)

    a = -(A ** 2 + A * E + F)
    b = -mu * (2 * A * B + B * E)
    c = -((mu * B) ** 2)

    # create polynomials & solve
    coeffs = np.zeros(9)
    coeffs[0] = c
    coeffs[3] = b
    coeffs[6] = a
    coeffs[8] = 1
    roots = poly.polyroots(coeffs)

    # find rhos from roots (which are r2 magnitude)
    rhos = A + mu * B / roots.real ** 3

    # filter out bad roots
    mask = (roots.imag == 0) & (rhos > 0) & (roots.real > 0)
    roots = roots[mask].real
    rhos = rhos[mask]
    
    return roots, rhos

# find dE with Newton's method for finding f & g
def find_dE(r2, r2dot, tau, n, a, x_guess, eps=1e-12):
    x = x_guess
    x_prev = 99999

    # helpful value to just calculate once, no need to keep calculating every loop
    val = np.dot(r2, r2dot) / (n * a ** 2)
    r2_mag = np.linalg.norm(r2)

    # iterate until change small enough
    while abs(x_prev - x) > eps:
        f_x = x - (1 - r2_mag / a) * np.sin(x) + val * (1 - np.cos(x)) - n * tau
        f_prime = 1 - (1 - r2_mag / a) * np.cos(x) + val * np.sin(x)
        x_prev = x
        x -= f_x / f_prime

    return x

# if flag is 0, then we use func
def fg(tau1, tau3, r2, r2dot, flag=0, mu=1):

    r2_mag = np.linalg.norm(r2)

    # if flag is > 0, then we assume it's of degree flag
    if flag > 0:

        u = mu / (r2_mag ** 3)
        z = np.dot(r2, r2dot) / (r2_mag ** 2)
        q = np.dot(r2dot, r2dot) / (r2_mag ** 2) - u

        # generate truncated polynomials
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
        
        # get the powers and compute
        powers1 = tau1 ** np.arange(flag + 1)
        powers3 = tau3 ** np.arange(flag + 1)
        
        f_series_terms = np.array(f_series_terms)
        g_series_terms = np.array(g_series_terms)

        return (np.dot(f_series_terms, powers1), np.dot(g_series_terms, powers1)), (np.dot(f_series_terms, powers3), np.dot(g_series_terms, powers3))
    
    elif flag == 0:
        
        # function version

        # find the orbital elements needed
        a = 1/(2 / r2_mag - np.dot(r2dot, r2dot) / mu)
        n = np.sqrt(mu / a ** 3)
        e = np.sqrt(1 - np.linalg.norm(np.cross(r2, r2dot)) ** 2 / (mu * a))
        
        # get the init guess
        val1 = np.dot(r2, r2dot) / (n * a ** 2)
        val2 = n * tau1 - val1
        sign = val1 * np.cos(val2) + (1 - r2_mag / a) * np.sin(val2)
        guess1 = n * tau1 + np.copysign(0.85 * e, sign) - val1

        # calculate dE
        dE1 = find_dE(r2, r2dot, tau1, n, a, guess1)
        f1 = 1 - a / r2_mag * (1 - np.cos(dE1))
        g1 = tau1 + 1 / n * (np.sin(dE1) - dE1)

        # same process for obs 3
        val1 = np.dot(r2, r2dot) / (n * a ** 2)
        val2 = n * tau3 - val1
        sign = val1 * np.cos(val2) + (1 - r2_mag / a) * np.sin(val2)
        guess3 = n * tau3 + np.copysign(0.85 * e, sign) - val1

        dE3 = find_dE(r2, r2dot, tau3, n, a, guess3)
        f3 = 1 - a / r2_mag * (1 - np.cos(dE3))
        g3 = tau3 + 1 / n * (np.sin(dE3) - dE3)

        return (f1, g1), (f3, g3)

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
    
    return dotdict(
        a=a,
        e=e,
        i=np.degrees(i) % 360,
        omega=np.degrees(omega) % 360,
        w=np.degrees(w) % 360,
        M0=np.degrees(M0) % 360
    )

# approximate E using Newton's method
# inverse of M = E - e*sin(E)
def get_E(M, e, eps=1e-004):
    # make guesses
    E_guess = M
    M_guess = E_guess - e*np.sin(E_guess)
    M_true = M

    # loop while we are not within epsilon
    while np.abs(M_guess - M_true) > eps:
        # our approximation for M
        M_guess = E_guess - e * np.sin(E_guess)
        # update our guess
        E_guess = E_guess - (M_true - (E_guess - e * np.sin(E_guess))) / (e * np.cos(E_guess) - 1)

    return E_guess

# the Method of Gauss!
def MoG(ras_obs, decs_obs, sun_vecs, times, epoch, debug=False, tolerance=1e-8):

    # get rho hats from ras & decs
    rho_hats = get_rho_hat(np.radians(ras_obs), np.radians(decs_obs)).T

    # find taus by multiplying by k
    taus = [(times[i] - times[1]) * CONSTS.k for i in range(3)]
    tau = taus[2] - taus[0]

    # calculate D values
    D0 = np.dot(rho_hats[0], np.cross(rho_hats[1], rho_hats[2]))
    # (R_i cross rho2) dot rho3
    D1 = [np.dot(np.cross(sun_vecs[i], rho_hats[1]), rho_hats[2]) for i in range(3)]
    # (rho1 cross R_i) dot rho3
    D2 = [np.dot(np.cross(rho_hats[0], sun_vecs[i]), rho_hats[2]) for i in range(3)]
    # rho1 dot (rho2 cross R_i)
    D3 = [np.dot(rho_hats[0], np.cross(rho_hats[1], sun_vecs[i])) for i in range(3)]

    # get roots/rhos & ask user if more than one solution
    roots, rhos = SEL([taus[0], taus[2], tau], sun_vecs[1], rho_hats[1], [D0, *D2])
    if len(rhos) > 1:
        print("ROOTS:", roots)
        print("RHOS:", rhos)
        index = int(input("Which root?"))
        rho2_mag = rhos[index]
    else:
        rho2_mag = rhos[0]

    # find the vectors 
    rho2 = rho_hats[1] * rho2_mag
    r2 = rho2 - sun_vecs[1]

    # we don't have r2dot yet, but we are only going up to the 2nd degree term
    (f1, g1), (f3, g3) = fg(taus[0], taus[2], r2, np.array([0., 0., 0.]), flag=3)

    # begin to iterate until rho2 magnitude change is small enough
    rho2_mag_prev = 9999
    i = 0
    while abs(rho2_mag_prev - rho2_mag) > tolerance:
        
        if debug:
            print("Prev RHO2 Mag", rho2_mag_prev)
            print("New RHO2 Mag", rho2_mag)
            print("Change", abs(rho2_mag_prev - rho2_mag))

        rho2_mag_prev = rho2_mag
        
        if debug:
            i += 1
            print("ITERATION", i)

        # find c's
        denom = f1 * g3 - g1 * f3
        c1 = g3 / denom
        c2 = -1
        c3 = -g1 / denom
        c = [c1, c2, c3]

        # find rho's and r's
        rho1 = np.dot(c, D1) / (c1 * D0) * rho_hats[0]
        rho2 = np.dot(c, D2) / (c2 * D0) * rho_hats[1]
        rho3 = np.dot(c, D3) / (c3 * D0) * rho_hats[2]

        r1 = rho1 - sun_vecs[0]
        r2 = rho2 - sun_vecs[1]
        r3 = rho3 - sun_vecs[2]

        rho2_mag = np.linalg.norm(rho2)

        # find r2dot from f & g
        d1 = -f3 / denom
        d3 = f1 / denom
        r2dot = d1 * r1 + d3 * r3

        # light travel correction
        t1 = times[0] - np.linalg.norm(rho1) / CONSTS.c
        t2 = times[1] - np.linalg.norm(rho2) / CONSTS.c
        t3 = times[2] - np.linalg.norm(rho3) / CONSTS.c

        # recalc taus
        taus[0] = (t1 - t2) * CONSTS.k
        taus[2] = (t3 - t2) * CONSTS.k
        tau = t3 - t1

        # update f & g
        (f1, g1), (f3, g3) = fg(taus[0], taus[2], r2, r2dot, flag=0)

    # convert to ecliptic
    r2_ecliptic = CONSTS.eq2ec_mat @ r2
    r2dot_ecliptic = CONSTS.eq2ec_mat @ r2dot
    
    # get orbital params
    o = get_orbit(r2_ecliptic, r2dot_ecliptic, t2, epoch)

    return r2_ecliptic, r2dot_ecliptic, o

# read input
obs = []
filename = input("FILENAME: ")
with open(filename) as f:
    # ignore commented lines
    lines = [line for line in f.readlines() if line[0] != "#"]
    
    tokens = lines[0].split()
    epoch = float(tokens[0])

    # read in each line
    for line in lines[1:]:
        
        tokens = line.split()
        jd = float(tokens[0])
        # get ra & dec in radians
        c = SkyCoord(tokens[1], tokens[2], unit=(u.hourangle, u.deg))
        ra = c.ra.rad
        dec = c.dec.rad

        # 6 token line is test input
        if len(tokens) == 6:
            sun_vector = np.array([float(tokens[3]), float(tokens[4]), float(tokens[5])])
            ra_err = 0
            dec_err = 0
            obscode="500"
        # otherwise get sun vector from horizons
        else:
            obscode = tokens[3]
            query = Horizons(id="10", location=obscode, epochs=jd, id_type="id")
            sun_vector = query.vectors(refplane="earth", aberrations="apparent")[0]
            sun_vector = np.array([sun_vector["x"], sun_vector["y"], sun_vector["z"]])

        obs.append(dict(
            ra=ra,
            dec=dec,
            jd=jd,
            sun_vector=sun_vector,
            obscode=obscode
        ))

# pick observations if more than 3
if len(obs) > 3:
    i, j, k = input("Which 3 observations (zero-indexing)?\n").replace(",", " ").split()
    i, j, k = sorted([int(i), int(j), int(k)])
    mog_obs = [obs[i], obs[j], obs[k]]
    del obs[k], obs[j], obs[i]
    obs = mog_obs + obs

# convert to pandas dataframe to get out the lists
obs = pd.DataFrame(obs)

sun_vecs = np.stack(obs["sun_vector"])
times = np.array(obs["jd"])
ras_obs = np.degrees(np.array(obs["ra"]))
decs_obs = np.degrees(np.array(obs["dec"]))

# get values and output
r2, r2dot, orbit = MoG(ras_obs, decs_obs, sun_vecs, times, epoch, debug=True)
print()
print("RESULTS:")
print("POSITION VECTOR (AU)    :", r2)
print("VELOCITY VECTOR (AU/day):", r2dot * CONSTS.k)
print("DIST TO EARTH:", r2 + sun_vecs[1])
print("ORBITAL PARAMS:")
pprint(orbit)