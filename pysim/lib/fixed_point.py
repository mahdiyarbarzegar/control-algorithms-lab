import numpy as np


class FixedPointVar:
    def __init__(self, bit_length=32, frac_length=15, wrap_style="saturate", signed=True, is_raw=False, value=0):
        if wrap_style != 'saturate' and wrap_style != 'wrap': raise ValueError(
            'wrap_style must be "saturate" or "wrap"')

        self.bit_length = bit_length
        self.frac_length = frac_length
        self.wrap_style = wrap_style
        self.signed = signed

        if self.signed:
            self.max = 2 ** (bit_length - 1) - 1
            self.min = -2 ** (bit_length - 1)
        else:
            self.max = 2 ** bit_length - 1
            self.min = 0

        if not is_raw:
            """Converts float/int to fixed-point representation."""
            int_val = int(round(value * (2 ** self.frac_length)))
        else:
            int_val = value

        self._val = self._apply_wrap(int_val, wrap_style)

    @classmethod
    def from_raw(cls, bit_length=32, frac_length=15, wrap_style="saturate", signed=True, value=0):
        return cls(bit_length, frac_length, wrap_style, signed, is_raw=True, value=value)

    @classmethod
    def from_float(cls, bit_length=32, frac_length=15, wrap_style="saturate", signed=True, value=0):
        return cls(bit_length, frac_length, wrap_style, signed, is_raw=False, value=value)

    def to_float(self):
        return float(self._val) / (2 ** self.frac_length)

    def get_raw(self):
        return self._val

    def _compare_ext_obj(self, obj):
        if self.bit_length != obj.bit_length:
            raise ValueError('bit_length mismatch')
        if self.frac_length != obj.frac_length:
            raise ValueError('frac_length mismatch')
        if self.signed != obj.signed:
            raise ValueError('signed mismatch')
        if self.wrap_style != obj.wrap_style:
            raise ValueError('wrap_style mismatch')

    def _apply_wrap(self, val, mode='saturate'):
        """
            mode: 'wrap' (Standard hardware) or 'saturate' (DSP style)
        """
        if mode == 'saturate':
            if val > self.max: return self.max
            if val < self.min: return self.min
            return val
        elif mode == 'wrap':
            mask = (1 << self.bit_length) - 1
            val &= mask
            # Handle sign bit for two's complement
            if self.signed and (val & (1 << (self.bit_length - 1))):
                val -= (1 << self.bit_length)
            return val
        else:
            raise ValueError('Invalid mode')

    def __add__(self, other):
        self._compare_ext_obj(other)
        return FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed,
                                      (self._val + other.get_raw()))

    def __sub__(self, other):
        self._compare_ext_obj(other)
        return FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed,
                                      (self._val - other.get_raw()))

    def __mul__(self, other):
        self._compare_ext_obj(other)

        raw_mul = self._val * other.get_raw()

        # rounding before shift
        if raw_mul >= 0:
            raw_res = (raw_mul + (1 << (self.frac_length - 1))) >> self.frac_length
        else:
            raw_res = -((-raw_mul + (1 << (self.frac_length - 1))) >> self.frac_length)

        return FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed, raw_res)

    def __truediv__(self, other):
        self._compare_ext_obj(other)

        den = other.get_raw()
        if den == 0:
            raise ZeroDivisionError("Fixed-point division by zero")

        num = self._val << self.frac_length

        if (num >= 0 and den > 0) or (num < 0 and den < 0):
            raw_res = (num + abs(den) // 2) // den
        else:
            raw_res = (num - abs(den) // 2) // den

        return FixedPointVar.from_raw(
            self.bit_length,
            self.frac_length,
            self.wrap_style,
            self.signed,
            raw_res
        )

    def __lshift__(self, n):
        return FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed, (self._val << n))

    def __ilshift__(self, n):
        self._val = self._apply_wrap(self._val << n, self.wrap_style)
        return self

    def __rshift__(self, n):
        return FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed, (self._val >> n))

    def __irshift__(self, n):
        self._val = self._apply_wrap(self._val >> n, self.wrap_style)
        return self

    def __gt__(self, other):
        self._compare_ext_obj(other)
        return self._val > other.get_raw()

    def __ge__(self, other):
        self._compare_ext_obj(other)
        return self._val >= other.get_raw()

    def __lt__(self, other):
        self._compare_ext_obj(other)
        return self._val < other.get_raw()

    def __le__(self, other):
        self._compare_ext_obj(other)
        return self._val <= other.get_raw()

    def __repr__(self):
        return f"{self._val / 2 ** self.frac_length} (raw: {self._val})"


class FixedPointMat:
    def __init__(self, mat, bit_length=32, frac_length=15, wrap_style="saturate", signed=True, is_raw=False):
        if wrap_style != 'saturate' and wrap_style != 'wrap':
            raise ValueError('wrap_style must be "saturate" or "wrap"')

        self.bit_length = bit_length
        self.frac_length = frac_length
        self.wrap_style = wrap_style
        self.signed = signed

        if self.signed:
            self.max = 2 ** (bit_length - 1) - 1
            self.min = -2 ** (bit_length - 1)
        else:
            self.max = 2 ** bit_length - 1
            self.min = 0

        self.shape = mat.shape
        rows, cols = mat.shape

        self._mat = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                if not is_raw:
                    self._mat[i, j] = FixedPointVar.from_float(bit_length, frac_length, wrap_style, signed, mat[i, j])
                else:
                    self._mat[i, j] = FixedPointVar.from_raw(bit_length, frac_length, wrap_style, signed, mat[i, j])

    @classmethod
    def from_raw(cls, mat, bit_length=32, frac_length=15, wrap_style="saturate", signed=True):
        return cls(mat, bit_length, frac_length, wrap_style, signed, is_raw=True)

    @classmethod
    def from_float(cls, mat, bit_length=32, frac_length=15, wrap_style="saturate", signed=True):
        return cls(mat, bit_length, frac_length, wrap_style, signed, is_raw=False)

    @property
    def T(self):
        res = np.empty((self.shape[1], self.shape[0]), dtype=object)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[j, i] = self._mat[i, j].get_raw()

        return FixedPointMat.from_raw(res, self.bit_length, self.frac_length, self.wrap_style, self.signed)

    @classmethod
    def zeros(cls, rows, cols, bit_length=32, frac_length=15, wrap_style="saturate", signed=True):
        mat = np.zeros((rows, cols), dtype=np.int64)

        return cls.from_raw(mat, bit_length, frac_length, wrap_style, signed)

    @classmethod
    def eye(cls, n, bit_length=32, frac_length=15, wrap_style="saturate", signed=True):
        mat = np.zeros((n, n), dtype=float)

        for i in range(n):
            mat[i, i] = 1.0

        return cls.from_float(mat, bit_length, frac_length, wrap_style, signed)

    def to_float(self):
        rows, cols = self.shape
        out = np.empty((rows, cols))

        for i in range(rows):
            for j in range(cols):
                out[i, j] = self._mat[i, j].to_float()

        return out

    def get_raw(self):
        rows, cols = self.shape
        out = np.empty((rows, cols))

        for i in range(rows):
            for j in range(cols):
                out[i, j] = self._mat[i, j].get_raw()

        return out

    def _compare_ext_obj(self, obj):
        if self.bit_length != obj.bit_length:
            raise ValueError('bit_length mismatch')
        if self.frac_length != obj.frac_length:
            raise ValueError('frac_length mismatch')
        if self.signed != obj.signed:
            raise ValueError('signed mismatch')
        if self.wrap_style != obj.wrap_style:
            raise ValueError('wrap_style mismatch')

    def shape(self):
        return self.shape[0], self.shape[1]

    def __add__(self, other):
        self._compare_ext_obj(other)

        if self.shape != other.shape:
            raise ValueError('shape mismatch')

        res = np.empty((self.shape[0], self.shape[1]), dtype=object)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i, j] = (self._mat[i, j] + other._mat[i, j]).get_raw()

        return FixedPointMat.from_raw(res, self.bit_length, self.frac_length, self.wrap_style, self.signed)

    def __sub__(self, other):
        self._compare_ext_obj(other)

        if self.shape != other.shape:
            raise ValueError('shape mismatch')

        res = np.empty((self.shape[0], self.shape[1]), dtype=object)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i, j] = (self._mat[i, j] - other._mat[i, j]).get_raw()

        return FixedPointMat.from_raw(res, self.bit_length, self.frac_length, self.wrap_style, self.signed)

    def __matmul__(self, other):
        self._compare_ext_obj(other)

        if self.shape[1] != other.shape[0]:
            raise ValueError('matrix multiplication shape mismatch')

        rows = self.shape[0]
        cols = other.shape[1]
        inner = self.shape[1]

        res = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                acc = FixedPointVar.from_raw(self.bit_length, self.frac_length, self.wrap_style, self.signed, 0)

                for k in range(inner):
                    acc = acc + self._mat[i, k] * other._mat[k, j]

                res[i, j] = acc.get_raw()

        return FixedPointMat.from_raw(res, self.bit_length, self.frac_length, self.wrap_style, self.signed)

    def __getitem__(self, item):
        return self._mat[item]

    def __setitem__(self, key, value):
        i, j = key

        if isinstance(value, FixedPointVar):
            self._mat[i, j] = value
        else:
            self._mat[i, j] = FixedPointVar.from_float(self.bit_length, self.frac_length, self.wrap_style, self.signed,
                                                       value)

    def inv_2x2(self):
        if self.shape != (2, 2):
            raise ValueError("inv_2x2 only supports 2x2 matrices")

        a = self._mat[0, 0]
        b = self._mat[0, 1]
        c = self._mat[1, 0]
        d = self._mat[1, 1]

        det = a * d - b * c

        if det.get_raw() == 0:
            raise ZeroDivisionError("singular matrix")

        zero = FixedPointVar.from_raw(
            self.bit_length,
            self.frac_length,
            self.wrap_style,
            self.signed,
            0
        )

        res = np.empty((2, 2), dtype=np.int64)

        res[0, 0] = (d / det).get_raw()
        res[0, 1] = ((zero - b) / det).get_raw()
        res[1, 0] = ((zero - c) / det).get_raw()
        res[1, 1] = (a / det).get_raw()

        return FixedPointMat.from_raw(
            res,
            self.bit_length,
            self.frac_length,
            self.wrap_style,
            self.signed
        )
