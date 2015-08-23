"""
Set of utility functions
"""

from jlgh import TOL_ANGLE, TOL_DIST
import numpy as np

def get_dot_cross_angle(v1,v2):
    """
    Wraps the operations of geting angles
    between vectors to reduce code repetition
    """
    cross = np.cross(v1,v2)
    dot = np.dot(v1,v2)
    angle = np.arctan2(np.linalg.norm(cross),dot)
    return dot, cross, angle

def nr2letter(n):
    """
    Maps natural numbers to
    ['0','1',...,'9','A','B',...,'Z']
    """
    if n < 10:
        return str(n)
    elif n <= (ord('Z') - ord('A') + 10):
        ch = chr(ord('A') + n - 10)
    else:
        raise NotImplementedError('Cannot convert number to character.'
                                  'You have to many sites, sorry!')

def lattice2nr((ix,iy,isite),size,nsites):
    """
    Convert a lattice point into a number corresponding
    to its position in the corresponding flattened (1D) array
    representing the lattice
    """
    return ((iy*size[0] + ix)*nsites + isite)

def nr2base26(nr):
    if isinstance(nr,str):
        nr = int(nr)
    if not isinstance(nr,int):
        raise TypeError('nr must be int or str')

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base26 = ''
    while nr:
        nr, i = divmod(nr,26)
        base26 = alphabet[i] + base26
    base26 = base26 or 'A'
    if len(base26) == 1:
        return 'A' + base26
    else:
        return base26

def nr2base16(nr):
    if isinstance(nr,str):
        nr = int(nr)
    if not isinstance(nr,int):
        raise TypeError('nr must be int or str')

    alphabet = '0123456789ABCDEF'
    base16 = ''
    while nr:
        nr, i = divmod(nr,16)
        base16 = alphabet[i] + base16
    return base16

def project_to_plane(array,v1,v2):
    a1 = np.dot(array,v1/np.linalg.norm(v1)) * v1 / np.linalg.norm(v1) ** 2
    a2 = np.dot(array,v2/np.linalg.norm(v2)) * v2 / np.linalg.norm(v2) ** 2
    return a1 + a2
