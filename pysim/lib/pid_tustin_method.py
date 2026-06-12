from lib.fixed_point import FixedPointVar as fpv


class PidCalcTustin:
    fp_wrap_style = 'saturate'

    def __init__(self, ts, kp, ki, kd, ka, n, fp_bl, fp_q, u_min, u_max):
        self.ts = ts
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ka = ka  # Anti-windup gain (typically between 0 and 1/ki)
        self.n = n

        self.bl = fp_bl
        self.q = fp_q

        self.ap = kp
        self.ai = ki * ts / 2
        self.ad = 2 * n * kd / (n * ts + 2)
        self.bd = (n * ts - 2) / (n * ts + 2)
        self.ap_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.ap)
        self.ai_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.ai)
        self.ad_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.ad)
        self.bd_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.bd)
        self.ka_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.ka)

        # Internal states
        self.e_last = [0.0, 0.0]
        self.e_last_fp = [
            fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 0),
            fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 0)
        ]
        self.u_last = [0.0, 0.0]
        self.u_last_fp = [
            fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 0),
            fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 0)
        ]

        self.u_min = u_min
        self.u_max = u_max
        self.sat_err = 0
        self.u_min_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, u_min)
        self.u_max_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, u_max)
        self.sat_err_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, self.sat_err)

    def calc_u(self, e):
        u_raw = (
                (1 - self.bd) * self.u_last[0]
                + self.bd * self.u_last[1]
                + self.ap * (e + (self.bd - 1) * self.e_last[0] - self.bd * self.e_last[1])
                + self.ai * (e + (self.bd + 1) * self.e_last[0] + self.bd * self.e_last[1])
                + self.ad * (e - 2 * self.e_last[0] + self.e_last[1])
                + self.sat_err
        )

        u_lmt = max(self.u_min, min(self.u_max, u_raw))

        self.sat_err = (u_lmt - u_raw) * self.ka

        self.e_last[1] = self.e_last[0]
        self.e_last[0] = e

        self.u_last[1] = self.u_last[0]
        self.u_last[0] = u_lmt

        return u_lmt

    def calc_u_fp(self, e):
        e_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, e)
        n_1_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 1)
        n_2_fp = fpv.from_float(self.bl, self.q, self.fp_wrap_style, True, 2)

        u_raw_fp = (
                (n_1_fp - self.bd_fp) * self.u_last_fp[0]
                + self.bd_fp * self.u_last_fp[1]
                + self.ap_fp * (e_fp + (self.bd_fp - n_1_fp) * self.e_last_fp[0] - self.bd_fp * self.e_last_fp[1])
                + self.ai_fp * (e_fp + (self.bd_fp + n_1_fp) * self.e_last_fp[0] + self.bd_fp * self.e_last_fp[1])
                + self.ad_fp * (e_fp - n_2_fp * self.e_last_fp[0] + self.e_last_fp[1])
                + self.sat_err_fp
        )

        u_lmt_fp = u_raw_fp

        if u_raw_fp > self.u_max_fp:
            u_lmt_fp = self.u_max_fp
        elif u_raw_fp < self.u_min_fp:
            u_lmt_fp = self.u_min_fp

        self.sat_err_fp = (u_lmt_fp - u_raw_fp) * self.ka_fp

        self.e_last_fp[1] = self.e_last_fp[0]
        self.e_last_fp[0] = e_fp

        self.u_last_fp[1] = self.u_last_fp[0]
        self.u_last_fp[0] = u_lmt_fp

        return u_lmt_fp.to_float()
