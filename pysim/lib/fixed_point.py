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
