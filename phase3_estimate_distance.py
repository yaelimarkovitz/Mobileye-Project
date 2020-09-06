import numpy as np
from math import sqrt

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
        
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose((curr_container))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    norm_pts = []
    for tfl in pts:
        norm_pts.append([(tfl[0] - pp[0])/focal,(tfl[1] - pp[1])/focal] )
    return np.array(norm_pts) 

def unnormalize(pts, focal, pp):
    unnorm_pts = []
    for tfl in pts:
        unnorm_pts.append([(tfl[0] * focal)+ pp[0],(tfl[1] * focal)+ pp[1]])
    return np.array(unnorm_pts)
    # transform normalized pixels into pixels using the focal length and principle point

def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM.EM[:3,:3]
    t = EM.EM[:,3]
    foe = [t[0]/t[2],t[1]/t[2]]
    return R,foe,t[2]

def rotate(pts, R):

    norm_rotate=[]
    for tfl in pts:
        p_vec = np.array([tfl[0],tfl[1],1])
        result = R.dot(p_vec)
        norm_rotate.append([result[0]/result[2],result[1]/result[2]])
    return np.array(norm_rotate)
    # rotate the points - pts using R

def find_distance_between_point_to_line(point,line_m,line_n):
    return abs((line_m*point[0]+line_n-point[1])/sqrt(line_m**2+1))

def find_corresponding_points(p, norm_pts_rot, foe):
    epipolar_line_m = (foe[1]-p[1])/(foe[0]-p[0])
    epipolar_line_n = (p[1]*foe[0] - foe[1]*p[0])/(foe[0]-p[0])
    distances = []
    for point in norm_pts_rot:
        distances.append(find_distance_between_point_to_line(point,epipolar_line_m,epipolar_line_n))

    print(distances)
    print(min(distances))
    return distances.index(min(distances)) , norm_pts_rot[distances.index(min(distances))]

    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    if (p_curr[0]-p_rot[0]>p_curr[1]-p_rot[1]):
        Z = (tZ* (foe[0]-p_rot[0]))/(p_curr[0]-p_rot[0])
    else:
        Z = (tZ* (foe[1]-p_rot[1]))/(p_curr[1]-p_rot[1])
    return Z