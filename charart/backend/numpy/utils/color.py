import numpy as np


illuminants = \
    {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
           '10': (1.111420406956693, 1, 0.3519978321919493),
           'R': (1.098466069456375, 1, 0.3558228003436005)},
     "B": {'2': (0.9909274480248003, 1, 0.8531327322886154),
           '10': (0.9917777147717607, 1, 0.8434930535866175),
           'R': (0.9909274480248003, 1, 0.8531327322886154)},
     "C": {'2': (0.980705971659919, 1, 1.1822494939271255),
           '10': (0.9728569189782166, 1, 1.1614480488951577),
           'R': (0.980705971659919, 1, 1.1822494939271255)},
     "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
             '10': (0.9672062750333777, 1, 0.8142801513128616),
             'R': (0.9639501491621826, 1, 0.8241280285499208)},
     "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
             '10': (0.9579665682254781, 1, 0.9092525159847462),
             'R': (0.9565317453467969, 1, 0.9202554587037198)},
     "D65": {'2': (0.95047, 1., 1.08883),
             '10': (0.94809667673716, 1, 1.0730513595166162),
             'R': (0.9532057125493769, 1, 1.0853843816469158)},
     "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
             '10': (0.9441713925645873, 1, 1.2064272211720228),
             'R': (0.9497220898840717, 1, 1.226393520724154)},
     "E": {'2': (1.0, 1.0, 1.0),
           '10': (1.0, 1.0, 1.0),
           'R': (1.0, 1.0, 1.0)}}

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

def _convert(matrix, arr):
    return arr @ matrix.T.astype(arr.dtype)

def get_xyz_coords(illuminant, observer, dtype : np.dtype = float):
    illuminant, observer = illuminant.upper(), observer.upper()
    try:
        return np.asarray(illuminants[illuminant][observer], dtype=dtype)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination '
                         f'(`{illuminant}`, `{observer}`)')

def rgb2xyz(rgb : np.ndarray) -> np.ndarray:
    arr = rgb.copy()
    mask = rgb > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] = arr[~mask] / 12.92
    return _convert(xyz_from_rgb, arr)

def xyz2rgb(xyz):
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] = arr[~mask] * 12.92
    return arr.clip(0, 1)

def xyz2lab(xyz, illuminant="D65", observer="2"):
    xyz_ref_white = get_xyz_coords(illuminant, observer, xyz.dtype)
    arr = xyz / xyz_ref_white
    mask = arr > 0.008856
    arr[mask] = np.power(arr[mask], 1 / 3)
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    L = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return np.stack([L, a, b], axis=-1)

def lab2xyz(lab, illuminant="D65", observer="2"):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    out = np.stack([x, y, z], axis=-1)
    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16. / 116.) / 7.787

    xyz_ref_white = get_xyz_coords(illuminant, observer, out.dtype)
    out = out * xyz_ref_white
    return out

def rgb2lab(rgb, illuminant="D65", observer="2"):
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)

def lab2rgb(lab, illuminant="D65", observer="2"):
    return xyz2rgb(lab2xyz(lab, illuminant, observer))


__all__ = ['rgb2lab', 'lab2rgb', 'rgb2xyz', 'xyz2rgb']
