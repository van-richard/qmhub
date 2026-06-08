import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from .dobject import DependObject, cache_update, invalidate_cache


def _unwrap_array(value):
    if isinstance(value, DependArray):
        value.update_cache()
        return value.array
    return value


def _wrap_array_result(cls, result):
    if isinstance(result, tuple):
        return tuple(_wrap_array_result(cls, item) for item in result)
    if isinstance(result, np.ndarray) and result.shape != ():
        return cls(result)
    return result


class DependArray(DependObject, NDArrayOperatorsMixin):

    __array_priority__ = 1000

    def __init__(self, data=None, **kwargs):
        if data is not None:
            self.array = np.asarray(data)
        else:
            self.array = None

        super().__init__(**kwargs)

    @classmethod
    def from_darray(cls, darray, index):
        ret = cls(darray[index], name=darray._name)

        if darray._func is not None:
            ret.add_dependency(darray)
        else:
            ret._dependants = darray._dependants

        return ret

    @cache_update
    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        if self._func is not None:
            raise NameError(f"Cannot set the value of <{self._name}> directly")
        self.array[index] = np.asarray(value, self.dtype)
        invalidate_cache(self)

    @cache_update
    def __bool__(self):
        return bool(self.array)

    @cache_update
    def __iter__(self):
        return iter(self.array)

    @cache_update
    def __reversed__(self):
        return reversed(self.array)

    def __array__(self, dtype=None, copy=None):
        self.update_cache()
        if copy is None:
            return np.asarray(self.array, dtype=dtype)
        return np.array(self.array, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outputs = kwargs.get("out", ())
        if outputs is None:
            outputs = ()
        elif not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_arrays = []
        for output in outputs:
            if isinstance(output, DependArray):
                if output._func is not None:
                    raise NameError(f"Cannot set the value of <{output._name}> directly")
                output_arrays.append(output.array)
            else:
                output_arrays.append(output)

        if output_arrays:
            kwargs["out"] = tuple(output_arrays)

        if method == "at" and inputs and isinstance(inputs[0], DependArray):
            if inputs[0]._func is not None:
                raise NameError(f"Cannot set the value of <{inputs[0]._name}> directly")
            result = getattr(ufunc, method)(*[_unwrap_array(value) for value in inputs], **kwargs)
            invalidate_cache(inputs[0])
            return result

        result = getattr(ufunc, method)(*[_unwrap_array(value) for value in inputs], **kwargs)

        if outputs:
            result_items = result if isinstance(result, tuple) else (result,)
            returned = []

            for output in outputs:
                if isinstance(output, DependArray):
                    invalidate_cache(output)
            for output, item in zip(outputs, result_items):
                if isinstance(output, DependArray):
                    returned.append(output)
                elif output is None:
                    returned.append(_wrap_array_result(self.__class__, item))
                else:
                    returned.append(item)

            if len(returned) == 1:
                return returned[0]
            return tuple(returned)

        return _wrap_array_result(self.__class__, result)

    @cache_update
    def __len__(self):
        return len(self.array)

    @cache_update
    def __repr__(self):
        if self.array.ndim > 0:
            return self.__class__.__name__ + repr(self.array)[len("array"):]
        return self.__class__.__name__ + "(" + repr(self.array) + ")"

    def __getattr__(self, attr):
        if attr == "array":
            raise AttributeError(attr)
        self.update_cache()
        return getattr(self.array, attr)

    def update_cache(self):
        if not self._cache_valid:
            if self._func is None:
                if self._dependencies:
                    for dobject in self._dependencies:
                        dobject.update_cache()
            else:
                self.array = np.ascontiguousarray(self._func(*self._dependencies, **self._kwargs))
            self._cache_valid = True

    @cache_update
    def copy(self):
        return _wrap_array_result(self.__class__, self.array.copy())

    @cache_update
    def astype(self, dtype):
        return _wrap_array_result(self.__class__, self.array.astype(dtype))

    @cache_update
    def tobytes(self, order='C'):
        ""
        return self.array.tobytes(order=order)

    def tostring(self, order='C'):
        return self.tobytes(order=order)
