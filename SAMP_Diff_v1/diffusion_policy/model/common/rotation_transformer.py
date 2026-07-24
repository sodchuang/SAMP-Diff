from typing import Union
import torch
import numpy as np
import functools

try:
    import pytorch3d.transforms as pt
except ImportError:
    pt = None


def _axis_angle_to_matrix_np(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    shape = x.shape
    y = Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix()
    return y.reshape(*shape[:-1], 3, 3)


def _matrix_to_axis_angle_np(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    shape = x.shape
    y = Rotation.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
    return y.reshape(*shape[:-2], 3)


def _matrix_to_rotation_6d_np(x: np.ndarray) -> np.ndarray:
    return x[..., :2, :].reshape(*x.shape[:-2], 6)


def _rotation_6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _fallback_transform_np(x: np.ndarray, from_rep: str, to_rep: str) -> np.ndarray:
    if from_rep == 'axis_angle':
        mat = _axis_angle_to_matrix_np(x)
    elif from_rep == 'rotation_6d':
        mat = _rotation_6d_to_matrix_np(x)
    elif from_rep == 'matrix':
        mat = x
    else:
        raise ImportError(
            f"RotationTransformer fallback only supports axis_angle, "
            f"rotation_6d, and matrix, got from_rep={from_rep!r}. "
            "Install pytorch3d for the full set of rotation conversions."
        )

    if to_rep == 'axis_angle':
        return _matrix_to_axis_angle_np(mat)
    if to_rep == 'rotation_6d':
        return _matrix_to_rotation_6d_np(mat)
    if to_rep == 'matrix':
        return mat
    raise ImportError(
        f"RotationTransformer fallback only supports axis_angle, "
        f"rotation_6d, and matrix, got to_rep={to_rep!r}. "
        "Install pytorch3d for the full set of rotation conversions."
    )


def _fallback_transform(x: Union[np.ndarray, torch.Tensor], from_rep: str, to_rep: str):
    is_numpy = isinstance(x, np.ndarray)
    dtype = x.dtype
    device = None
    if is_numpy:
        x_np = x
    else:
        device = x.device
        x_np = x.detach().cpu().numpy()

    y = _fallback_transform_np(x_np, from_rep, to_rep)
    if is_numpy:
        return y.astype(dtype, copy=False)
    return torch.from_numpy(y).to(device=device, dtype=dtype)

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        self.from_rep = from_rep
        self.to_rep = to_rep
        self.use_pytorch3d = pt is not None
        forward_funcs = list()
        inverse_funcs = list()

        if self.use_pytorch3d:
            if from_rep != 'matrix':
                funcs = [
                    getattr(pt, f'{from_rep}_to_matrix'),
                    getattr(pt, f'matrix_to_{from_rep}')
                ]
                if from_convention is not None:
                    funcs = [functools.partial(func, convention=from_convention)
                        for func in funcs]
                forward_funcs.append(funcs[0])
                inverse_funcs.append(funcs[1])

            if to_rep != 'matrix':
                funcs = [
                    getattr(pt, f'matrix_to_{to_rep}'),
                    getattr(pt, f'{to_rep}_to_matrix')
                ]
                if to_convention is not None:
                    funcs = [functools.partial(func, convention=to_convention)
                        for func in funcs]
                forward_funcs.append(funcs[0])
                inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        if not self.use_pytorch3d:
            return _fallback_transform(x, self.from_rep, self.to_rep)
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        if not self.use_pytorch3d:
            return _fallback_transform(x, self.to_rep, self.from_rep)
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
