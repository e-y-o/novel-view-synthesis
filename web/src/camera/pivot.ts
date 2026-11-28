/**
 * Half-float helpers and the pivot choice, mirroring `choose_pivot` in `src/nvs/core/camera.py`.
 *
 * The preview depth arrives as raw little-endian float16 (`depth.f16`); `halfToFloat` decodes one value.
 */

/** Decode one IEEE 754 binary16 bit pattern (0..0xFFFF) to a number. Handles subnormals, ±0, ±Inf and NaN. */
export function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24; // zero and subnormals
  if (exp === 0x1f) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Round a number to the nearest binary16 bit pattern (ties to even); overflow becomes ±Inf. */
export function floatToHalf(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a === Infinity) return sign | 0x7c00;
  if (a < 2 ** -14) {
    // subnormal range: multiples of 2^-24, round half to even
    return sign | roundHalfEven(a * 2 ** 24);
  }
  let exp = Math.floor(Math.log2(a));
  if (2 ** exp > a) exp -= 1; // guard against log2 rounding up
  if (2 ** (exp + 1) <= a) exp += 1;
  const mant = roundHalfEven((a / 2 ** exp - 1) * 1024);
  let e = exp + 15;
  let m = mant;
  if (m === 1024) {
    m = 0;
    e += 1;
  }
  if (e >= 0x1f) return sign | 0x7c00;
  return sign | (e << 10) | m;
}

function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Decode a whole `Uint16Array` of half-float bit patterns. */
export function decodeHalfArray(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) out[i] = halfToFloat(bits[i] as number);
  return out;
}

const PIVOT_PERCENTILE = 20;
const PIVOT_MAX_SAMPLES = 1_000_000;

/**
 * Orbit pivot depth [m]: `min(p20 of finite non-sky depth, cap)`; `cap` if no valid pixel.
 *
 * @param depth metres, row-major; a `Float32Array`, or a `Uint16Array` of half-float bit patterns
 *   (the raw `depth.f16` of the preview).
 * @param sky one byte per pixel, non-zero = sky (`sky.u8`: 255 sky, 0 otherwise); same length as `depth`.
 * @param cap upper bound in metres (default 5).
 *
 * The percentile is linear-interpolated like numpy's default, over a deterministic strided subsample when
 * there are more than a million valid pixels (the same rule as Python). On the preview depth the result
 * matches the server's `pivot_z` to within about 1 %.
 */
export function choosePivot(depth: Float32Array | Uint16Array, sky: Uint8Array, cap = 5): number {
  if (depth.length !== sky.length) throw new RangeError("depth and sky must have the same length");
  const isHalf = depth instanceof Uint16Array;
  const valid: number[] = [];
  for (let i = 0; i < depth.length; i += 1) {
    if (sky[i] !== 0) continue;
    const raw = depth[i] as number;
    const d = isHalf ? halfToFloat(raw) : raw;
    if (Number.isFinite(d) && d > 0) valid.push(d);
  }
  if (valid.length === 0) return cap;
  let values = Float64Array.from(valid);
  if (values.length > PIVOT_MAX_SAMPLES) {
    const stride = Math.floor(values.length / PIVOT_MAX_SAMPLES);
    values = values.filter((_, i) => i % stride === 0);
  }
  values.sort();
  const pos = (PIVOT_PERCENTILE / 100) * (values.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, values.length - 1);
  const p = (values[lo] as number) + ((values[hi] as number) - (values[lo] as number)) * (pos - lo);
  return Math.min(p, cap);
}
