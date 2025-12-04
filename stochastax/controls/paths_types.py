from dataclasses import dataclass
from typing import Sequence, override
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Path:
    path: jax.Array
    interval: tuple[float, float]

    @property
    def num_timesteps(self) -> int:
        if self.path.ndim == 2:
            return self.path.shape[0]
        elif self.path.ndim == 3:
            return self.path.shape[1]
        else:
            raise ValueError(f"Path must be 2D or 3D. Got {self.path.ndim}D.")

    @property
    def ambient_dimension(self) -> int:
        if self.path.ndim == 2:
            return self.path.shape[1]
        elif self.path.ndim == 3:
            return self.path.shape[2]
        else:
            raise ValueError(f"Path must be 2D or 3D. Got {self.path.ndim}D.")

    def split_at_time(self, split_indices: int | Sequence[int]) -> list["Path"]:
        """
        Split the path into a list of paths at the given indices.
        """
        time_dim = 0 if self.path.ndim == 2 else 1

        if isinstance(split_indices, int):
            split_indices = [split_indices]
        full_indices = [0] + list(split_indices) + [self.path.shape[time_dim]]

        paths = []
        for i, j in zip(full_indices[:-1], full_indices[1:]):
            time_slice = slice(i, j)
            if self.path.ndim == 3:
                paths.append(self[slice(None), time_slice])
            else:
                paths.append(self[time_slice])
        return paths

    def __getitem__(self, idx: int | slice | tuple) -> "Path":
        sub_path = self.path[idx]

        original_time_dim = 1 if self.path.ndim == 3 else 0
        indices = idx if isinstance(idx, tuple) else (idx,)

        time_dim_indexer = None
        if self.path.ndim == 2 and len(indices) > 0:
            time_dim_indexer = indices[0]
        elif self.path.ndim == 3 and len(indices) > 1:
            time_dim_indexer = indices[1]

        new_interval = self.interval
        time_len = self.path.shape[original_time_dim]

        if isinstance(time_dim_indexer, slice):
            start, stop, stride = time_dim_indexer.indices(time_len)
            if stride != 1:
                raise ValueError("Slicing with a step is not supported for Path objects.")
            g0, g1 = self.interval[0] + start, self.interval[0] + stop
            new_interval = (g0, g1)
        elif isinstance(time_dim_indexer, int):
            if time_dim_indexer < 0:
                time_dim_indexer += time_len
            g0 = g1 = self.interval[0] + time_dim_indexer
            new_interval = (g0, g1)

        return Path(sub_path, new_interval)

    @override
    def __str__(self) -> str:
        string = f"""{self.__class__.__name__}(
    interval={self.interval},
    num_timesteps={self.num_timesteps},
    ambient_dimension={self.ambient_dimension},
    path_shape={self.path.shape}
)"""
        return string

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            raise NotImplementedError(f"Cannot compare Path with {type(other)}.")

        return bool(jnp.allclose(self.path, other.path)) and self.interval == other.interval

    def __add__(self, other: "Path") -> "Path":
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError(
                f"Paths must have the same ambient dimension. Got {self.ambient_dimension} and {other.ambient_dimension}."
            )
        if self.interval[1] != other.interval[0]:
            raise ValueError(
                f"Paths must have contiguous intervals. Got {self.interval} and {other.interval}."
            )
        if self.path.ndim != other.path.ndim:
            raise ValueError(
                f"Paths must have the same number of dimensions. Got {self.path.ndim} and {other.path.ndim}."
            )

        time_axis = 1 if self.path.ndim == 3 else 0
        new_path = jnp.concatenate([self.path, other.path], axis=time_axis)
        new_interval = (self.interval[0], other.interval[1])
        return Path(path=new_path, interval=new_interval)


jax.tree_util.register_pytree_node(
    Path,
    lambda p: ((p.path,), (p.interval,)),
    lambda aux, children: Path(children[0], *aux),
)


def pathify(stream: jax.Array) -> Path:
    """
    Converts a JAX array of a data stream into a Path object.
    """
    if stream.ndim == 2:
        interval = (0, stream.shape[0])
        return Path(path=stream, interval=interval)
    elif stream.ndim == 3:
        interval = (0, stream.shape[1])
        return Path(path=stream, interval=interval)
    else:
        raise ValueError(f"Stream must be a 2D or 3D array. Got shape {stream.shape}.")
