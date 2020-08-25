"""
Routines for generating Monte Carlo moves of varying types
"""
import numpy as np
import math as math
from util.bead_selection import *
from util.linalg import *
from .calc_density import calc_density


def mc_move(polymers, epigenmarks, density, num_epigenmark, num_polymers, mcmove, field):
    mc_move_type = mcmove.id
    # MC move type 0: Crank-shaft move
    if mc_move_type == 0:
        for i_move in range(mcmove.num_per_cycle):
            for poly in polymers:
                crank_shaft_move(poly, epigenmarks, density, mcmove, field)
                mcmove.num_attempt += 1

    # MC move type 1: End pivot move
    elif mc_move_type == 1:
        window_size = mcmove.amp_bead
        for i_move in range(mcmove.num_per_cycle):
            for poly in polymers:
                end_pivot_move(poly, epigenmark, density, mcmove, field, window_size)
                mcmove.num_attempt += 1

    # MC move type 2: Slide move
    elif mc_move_type == 2:
        pass

    # MC move type 3: Slide move
    elif mc_move_type == 3:
        pass

    # MC move type 4: Epigenetic protein binding move
    elif mc_move_type == 4:
        pass

    return


def crank_shaft_move(polymer, epigenmark, density, mcmove, field):

    # Select ind0 and indf for the crank-shaft move
    delta_ind = min(np.random.randint(2, mcmove.amp_bead), polymer.num_beads)
    ind0 = np.random.randint(polymer.num_beads - delta_ind + 1)
    indf = ind0 + delta_ind

    # Generate the rotation matrix and vector around the vector between bead ind0 and indf
    rot_angle = mcmove.amp_move * (np.random.rand() - 0.5)

    if ind0 == (indf + 1):
        delta_t3 = polymer.t_3[ind0, :]
    else:
        delta_t3 = polymer.r[indf - 1, :] - polymer.r[ind0, :]
        delta_t3 /= np.linalg.norm(delta_t3)

    r_ind0 = polymer.r[ind0, :]

    rot_matrix = np.zeros((3, 3), 'd')

    rot_matrix[0, 0] = delta_t3[0] ** 2. + (delta_t3[1] ** 2 + delta_t3[2] ** 2) * math.cos(rot_angle)
    rot_matrix[0, 1] = delta_t3[0] * delta_t3[1] * (1 - math.cos(rot_angle)) - delta_t3[2] * math.sin(rot_angle)
    rot_matrix[0, 2] = delta_t3[0] * delta_t3[2] * (1 - math.cos(rot_angle)) + delta_t3[1] * math.sin(rot_angle)

    rot_matrix[1, 0] = delta_t3[0] * delta_t3[1] * (1 - math.cos(rot_angle)) + delta_t3[2] * math.sin(rot_angle)
    rot_matrix[1, 1] = delta_t3[1] ** 2. + (delta_t3[0] ** 2 + delta_t3[2] ** 2) * math.cos(rot_angle)
    rot_matrix[1, 2] = delta_t3[1] * delta_t3[2] * (1 - math.cos(rot_angle)) - delta_t3[0] * math.sin(rot_angle)

    rot_matrix[2, 0] = delta_t3[0] * delta_t3[2] * (1 - math.cos(rot_angle)) - delta_t3[1] * math.sin(rot_angle)
    rot_matrix[2, 1] = delta_t3[1] * delta_t3[2] * (1 - math.cos(rot_angle)) + delta_t3[0] * math.sin(rot_angle)
    rot_matrix[2, 2] = delta_t3[2] ** 2. + (delta_t3[0] ** 2 + delta_t3[1] ** 2) * math.cos(rot_angle)

    rot_vector = np.cross(r_ind0, delta_t3) * math.sin(rot_angle)
    rot_vector[0] += (r_ind0[0] * (1 - delta_t3[0] ** 2)
                      - delta_t3[0]*(r_ind0[1] * delta_t3[1] + r_ind0[2] * delta_t3[2])) * (1 - math.cos(rot_angle))
    rot_vector[1] += (r_ind0[1] * (1 - delta_t3[1] ** 2)
                      - delta_t3[1]*(r_ind0[0] * delta_t3[0] + r_ind0[2] * delta_t3[2])) * (1 - math.cos(rot_angle))
    rot_vector[2] += (r_ind0[2] * (1 - delta_t3[2] ** 2)
                      - delta_t3[2]*(r_ind0[0] * delta_t3[0] + r_ind0[1] * delta_t3[1])) * (1 - math.cos(rot_angle))

    # Generate the trial positions and orientations

    r_poly_trial = np.zeros((indf - ind0, 3), 'd')
    t3_poly_trial = np.zeros((indf - ind0, 3), 'd')
    for i_bead in range(ind0, indf):
        r_poly_trial[i_bead - ind0, :] = rot_vector + np.matmul(rot_matrix, polymer.r[i_bead, :])
        t3_poly_trial[i_bead - ind0, :] = np.matmul(rot_matrix, polymer.t_3[i_bead, :])

    # Calculate the change in energy
    density_poly, index_xyz = calc_density(polymer.r[ind0:indf, :],
                                           polymer.states, ind0, indf, field)
    density_poly_trial, index_xyz_trial = calc_density(r_poly_trial,
                                                       polymer.states,
                                                       ind0, indf, field)
    delta_density_poly_total = np.concatenate((density_poly_trial, -density_poly))
    delta_index_xyz_total = np.concatenate((index_xyz_trial, index_xyz)).astype(int)
    delta_density, delta_index_xyz = combine_repeat(delta_density_poly_total, delta_index_xyz_total)

    delta_energy_epigen = 0
    for i, epi_info in enumerate(epigenmark):
        delta_energy_epigen += 0.5 * epi_info.interaction_energy * np.sum(
            (delta_density[:, i + 1] + density[delta_index_xyz, i + 1]) ** 2
            - density[delta_index_xyz, i + 1] ** 2)

    delta_energy_poly = calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, ind0, indf)
    delta_energy = delta_energy_poly + delta_energy_epigen

    # Determine acceptance of trial based on Metropolis criterion
    if np.random.rand() < math.exp(-delta_energy):
        polymer.r[ind0:indf, :] = r_poly_trial
        polymer.t_3[ind0:indf, :] = t3_poly_trial
        density[delta_index_xyz, :] += delta_density
        mcmove.num_success += 1


def calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, ind0, indf):
    """
    Calculate the change in polymer energy for the trial

    :param r_poly:
    :param r_poly_trial:
    :param t3_poly:
    :param t3_poly_trial:
    :param polymer:
    :param i_poly:
    :param ind0:
    :param indf:
    :return:
    """

    delta_energy_poly = 0

    # Calculate contribution to polymer energy at the ind0 position
    if ind0 != 0:

        delta_r_trial = r_poly_trial[0, :] - polymer.r[ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, polymer.t_3[ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * polymer.t_3[ind0 - 1, :]

        delta_r = polymer.r[ind0, :] - polymer.r[ind0 - 1, :]
        delta_r_par = np.dot(delta_r, polymer.t_3[ind0 - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer.t_3[ind0 - 1, :]

        bend_vec_trial = (t3_poly_trial[0, :] - polymer.t_3[ind0 - 1, :]
                          - polymer.eta * delta_r_perp_trial)
        bend_vec = (polymer.t_3[ind0, :] - polymer.t_3[ind0 - 1, :]
                          - polymer.eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer.eps_par * (delta_r_par_trial - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer.eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer.eps_par * (delta_r_par - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp, delta_r_perp))

    # Calculate contribution to polymer energy at the indf position
    if indf != polymer.num_beads:

        delta_r_trial = polymer.r[indf, :] - r_poly_trial[indf - ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, t3_poly_trial[indf - ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * t3_poly_trial[indf - ind0 - 1, :]

        delta_r = polymer.r[indf, :] - polymer.r[indf - 1, :]
        delta_r_par = np.dot(delta_r, polymer.t_3[indf - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer.t_3[indf - 1, :]

        bend_vec_trial = (polymer.t_3[indf, :] - t3_poly_trial[indf - ind0 - 1, :]
                          - polymer.eta * delta_r_perp_trial)
        bend_vec = (polymer.t_3[indf, :] - polymer.t_3[indf - 1, :]
                    - polymer.eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer.eps_par * (delta_r_par_trial - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer.eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer.eps_par * (delta_r_par - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp, delta_r_perp))

    return delta_energy_poly


def combine_repeat(a, idx):
    """
    Combine the repeat entries in an array

    :param a:
    :return: a_combine, idx
    """""
    #    a = np.array([[11, 2], [11, 3], [13, 4], [10, 10], [10, 1]])
    #    b = a[np.argsort(a[:, 0])]
    #    grps, idx = np.unique(b[:, 0], return_index=True)
    #    counts = np.add.reduceat(b[:, 1:], idx)
    #    print(np.column_stack((grps, counts)))

    b = idx[np.argsort(idx)]
    grps, idx = np.unique(b, return_index=True)
    counts = np.add.reduceat(a, idx)
    a_with_index = np.column_stack((grps, counts))
    a_combine = a_with_index[:, 1:]
    idx_combine = a_with_index[:, 0].astype(int)

    return a_combine, idx_combine


def assess_energy_change(polymer, epigenmark, density, ind0, indf, field, r_poly_trial, t3_poly_trial):
    """
    Assess energy change and acceptance of MC move
    """
   
    density_poly, index_xyz = calc_density(polymer.r[ind0:indf, :], polymer.epigen_bind,
                                           ind0, indf, field)
    density_poly_trial, index_xyz_trial = calc_density(r_poly_trial, polymer.epigen_bind,
                                                       ind0, indf, field)
    delta_density_poly_total = np.concatenate((density_poly_trial, -density_poly))
    delta_index_xyz_total = np.concatenate((index_xyz_trial, index_xyz)).astype(int)
    delta_density, delta_index_xyz = combine_repeat(delta_density_poly_total, delta_index_xyz_total)

    for i, epi_info in enumerate(epigenmark):
        delta_energy_epigen += 0.5 * epi_info.interaction_energy * np.sum(
            (delta_density[:, i + 1] + density[delta_index_xyz, i + 1]) ** 2
            - density[delta_index_xyz, i + 1] ** 2)

    delta_energy_poly = calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, ind0, indf)
    delta_energy = delta_energy_poly + delta_energy_epigen

    return delta_index_xyz, delta_density, delta_energy


def assess_move_acceptance(polymer, ind0, indf, density, mcmove, r_poly_trial, t3_poly_trial, delta_index_xyz, delta_density, delta_energy):
    """
    Determine move acceptance based on Metropolis criterion
    """
    if delta_energy < 0:    # Favorable move
        prob = 1 
    else:                   # Unfavorable move
        try:
            prob = math.exp(-delta_energy) 
        except OverflowError:
            prob = 0 
    if np.random.rand() < prob:
        polymer.r[ind0:indf, :] = r_poly_trial
        polymer.t_3[ind0:indf, :] = t3_poly_trial
        density[delta_index_xyz, :] += delta_density
        mcmove[1].num_success += 1

    return


def end_pivot_move(polymer, epigenmark, density, mcmove, field, window_size):
    """
    Randomly rotate segment from one end of the polymer. 

    Parameters
    ----------
    polymer : Polymer
        Object containing polymer properties
    epigenmark : epigenmark object
        Object contianing epigenetic mark properties
    density : float
        Mass density of the polymer
    mcmove : mcmove object
        Object describing characteristics of the MC move (e.g. amplitude)
    field : Field
        Object describing properties of the energy field
    window_size : int
        Maximum number of nucleosomes to roate in any one MC move
    """
    
    rot_angle = mcmove.amp_move * (np.random.rand() - 0.5)
    num_beads = polymer.num_beads
    select_window = num_beads - 1
    all_beads = np.arange(0, num_beads, 1)

    if np.random.randint(0, 2) == 0:    # rotate LHS of polymer
        ind0 = 0
        pivot_point = indf = select_bead_from_left(select_window, num_beads)
        base_point = pivot_point + 1
    else:                               # rotate RHS of polymer
        pivot_point = ind0 = select_bead_from_right(select_window, num_beads)
        indf = num_beads-1
        base_point = pivot_point - 1
    
    r_points = np.ones((4, indf-ind0+1)
    r_points[0:3, :] = polymer.r[ind0:indf+1, :].T
    t3_points = np.ones((4, indf-ind0+1))
    t3_points[0:3, :] = polymer.t_3[ind0:indf+1, :].T
    
    r_pivot_point = polymer.r[pivot_point, :]
    r_base_point = polymer.r[base_point, :]
    
    rot_matrix = arbitrary_axis_rotation(r_pivot_point, r_base_point, rot_angle)
    r_poly_trial = np.matmul(rot_matrix, r_points)      # generate trial coordinates
    r_poly_trial = r_poly_trial[0:3,:].T
    t3_poly_trial = np.matmul(rot_matrix, t3_points)    # generate trial tangents
    t3_poly_trial = t3_poly_trial[0:3,:].T
            
    # assess move by Metropolis Criterion
    delta_index_xyz, delta_density, delta_energy = assess_energy_change(polymer, epigenmark, density, ind0, indf+1, field, r_poly_trial, t3_poly_trial)
    assess_move_acceptance(polymer, ind0, indf+1, density, mcmove, r_poly_trial, t3_poly_trial, delta_index_xyz, delta_density, delta_energy)

    return


def slide_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field, window_size):
    """
    Randomly translate a segment of the polymer.

    Parameters
    ----------
    polymer : Polymer
        Object containing polymer properties
    epigenmark : epigenmark object
        Object contianing epigenetic mark properties
    density : float
        Mass density of the polymer
    num_epigenmark : int
        Number of epigenetic marks included in the simulation
    i_poly : int
        Index of individual polymer in polymer object
    mcmove : mcmove object
        Object describing characteristics of the MC move (e.g. amplitude)
    field : Field
        Object describing properties of the energy field
    window_size : int
        Maximum number of nucleosomes to translate in any one MC move
    """

    active_poly = polymer[i_poly]   # isolate affected by move
    active_move = mcmove[2]         # isolate active move
    
    # Set move amolitude
    translation_amp = active_move.amp_move * (np.random.rand())

    # Partition translation move into x, y, z components
    rand_z = np.random.rand()
    translation_z = translation_amp * rand_z    
    rand_angle = np.random.rand() * 2 * np.pi
    translation_y = np.sqrt(1 - rand_z) * translation_amp * np.sin(rand_angle)
    translation_z = np.sqrt(1 - rand_z) * translation_amp * np.cos(rand_angle)

    # Sepect a segment of nucleosomes on which to apply the move
    num_beads = active_poly.num_beads
    ind0 = np.random.randint(num_beads)
    select_window = np.amin(np.array([num_beads-ind0, ind0, window_size]))
    indf = select_bead_from_point(select_window, num_beads, ind0)

    translation_mat = np.identity(4)
    translation_mat[0, 3] = translation_x
    translation_mat[1, 3] = translation_y
    translation_mat[2, 3] = translation_z
 
    r_points = np.ones((4, indf-ind0+1)
    r_points[0:3, :] = active_poly.r_poly[ind0:indf+1, :].T
    t3_poly = active_poly.t3_poly[ind0:indf+1, :]
    
    r_poly_trial = translation_mat @ r_points
    
    # assess move by Metropolis Criterion
    delta_index_xyz, delta_density, delta_energy = assess_energy_change(polymer, epigenmark, density, num_epigenmark, i_poly, ind0, indf+1, field, r_poly_trial, t3_poly)
    assess_move_acceptance(polymer, i_poly, ind0, indf+1, density, mcmove, r_poly_trial, t3_poly, delta_index_xyz, delta_density, delta_energy)

   return

