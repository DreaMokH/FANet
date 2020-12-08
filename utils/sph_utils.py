import numpy as np
import cv2
import math as m
from pylab import *
import torch


FACE_B = 0
FACE_D = 1
FACE_F = 2
FACE_L = 3
FACE_R = 4
FACE_T = 5


def rotx(ang):
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])


def roty(ang):
    return np.array([[np.cos(ang), 0, np.sin(ang)],
                     [0, 1, 0],
                     [-np.sin(ang), 0, np.cos(ang)]])


def rotz(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0],
                     [0, 0, 1]])


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def xy2angle(XX, YY, im_w, im_h):
    _XX = 2*(XX+0.5)/float(im_w)-1
    _YY = 1-2*(YY+0.5)/float(im_h)

    theta = _XX*m.pi
    phi = _YY*m.pi/2

    return theta, phi


def to_3dsphere(theta, phi, R): 
    x = R*torch.cos(phi)*torch.cos(theta)
    y = R*torch.sin(phi)
    z = R*torch.cos(phi)*torch.sin(theta)
    return x, y, z


def pruned_inf(angle):
    float_err = 10e-9
    angle[angle == 0.0] = float_err
    angle[angle == m.pi] = m.pi-float_err
    angle[angle == -m.pi] = -m.pi+float_err
    angle[angle == m.pi/2] = m.pi/2-float_err
    angle[angle == -m.pi/2] = -m.pi/2+float_err
    return angle


def over_pi(angle):
    while(angle > np.pi):
        angle -= 2*np.pi
    while(angle < -np.pi):
        angle += 2*np.pi
    return angle


def get_face(x, y, z, face_map):
    eps = 10e-9

    max_arr = torch.max(torch.max(torch.abs(x), torch.abs(y)), torch.abs(z))

    x_faces = max_arr-torch.abs(x) < eps
    y_faces = max_arr-torch.abs(y) < eps
    z_faces = max_arr-torch.abs(z) < eps

    face_map[(x >= 0) & x_faces] = FACE_F
    face_map[(x <= 0) & x_faces] = FACE_B
    face_map[(y >= 0) & y_faces] = FACE_T
    face_map[(y <= 0) & y_faces] = FACE_D
    face_map[(z >= 0) & z_faces] = FACE_R
    face_map[(z <= 0) & z_faces] = FACE_L
    return face_map


def face_to_cube_coord(face_gr, x, y, z):

    direct_coord = torch.zeros((face_gr.shape[0], face_gr.shape[1], 3))
    direct_coord[:, :, 0][face_gr == FACE_F] = z[face_gr == FACE_F]
    direct_coord[:, :, 1][face_gr == FACE_F] = y[face_gr == FACE_F]
    direct_coord[:, :, 2][face_gr == FACE_F] = x[face_gr == FACE_F]

    direct_coord[:, :, 0][face_gr == FACE_B] = -z[face_gr == FACE_B]
    direct_coord[:, :, 1][face_gr == FACE_B] = y[face_gr == FACE_B]
    direct_coord[:, :, 2][face_gr == FACE_B] = x[face_gr == FACE_B]

    direct_coord[:, :, 0][face_gr == FACE_T] = z[face_gr == FACE_T]
    direct_coord[:, :, 1][face_gr == FACE_T] = -x[face_gr == FACE_T]
    direct_coord[:, :, 2][face_gr == FACE_T] = y[face_gr == FACE_T]

    direct_coord[:, :, 0][face_gr == FACE_D] = z[face_gr == FACE_D]
    direct_coord[:, :, 1][face_gr == FACE_D] = x[face_gr == FACE_D]
    direct_coord[:, :, 2][face_gr == FACE_D] = y[face_gr == FACE_D]

    direct_coord[:, :, 0][face_gr == FACE_R] = -x[face_gr == FACE_R]
    direct_coord[:, :, 1][face_gr == FACE_R] = y[face_gr == FACE_R]
    direct_coord[:, :, 2][face_gr == FACE_R] = z[face_gr == FACE_R]

    direct_coord[:, :, 0][face_gr == FACE_L] = x[face_gr == FACE_L]
    direct_coord[:, :, 1][face_gr == FACE_L] = y[face_gr == FACE_L]
    direct_coord[:, :, 2][face_gr == FACE_L] = z[face_gr == FACE_L]

    x_oncube = (direct_coord[:, :, 0]/torch.abs(direct_coord[:, :, 2])+1)/2
    y_oncube = (-direct_coord[:, :, 1]/torch.abs(direct_coord[:, :, 2])+1)/2

    return x_oncube, y_oncube


def norm_to_cube(_out_coord, w):
    out_coord = _out_coord*(w-1)
    out_coord[out_coord < 0.] = 0.
    out_coord[out_coord > (w-1)] = (w-1)
    return out_coord
