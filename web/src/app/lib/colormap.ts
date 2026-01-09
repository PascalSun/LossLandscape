/**
 * Professional Coolwarm colormap for loss landscape visualizations
 * Based on matplotlib's coolwarm diverging colormap, optimized for loss values
 * 
 * This colormap is:
 * - Perceptually uniform: color changes correspond to value changes
 * - Colorblind-friendly: distinguishable for common color vision deficiencies
 * - Intuitive: blue (low loss, good) -> white (medium) -> red (high loss, bad)
 * - Widely used in scientific visualization
 * 
 * Color scheme:
 * - t=0 (low loss): Deep blue RGB(58, 76, 192) - indicates good performance
 * - t=0.5 (medium loss): White RGB(255, 255, 255) - neutral middle ground
 * - t=1 (high loss): Deep red RGB(180, 4, 38) - indicates poor performance
 */

// Coolwarm colormap key points (based on matplotlib implementation)
// These are carefully chosen RGB values for perceptual uniformity
const COOLWARM_KEYPOINTS = [
  { t: 0.0, r: 0.229, g: 0.298, b: 0.753 },  // Deep blue
  { t: 0.25, r: 0.303, g: 0.533, b: 0.956 }, // Light blue
  { t: 0.5, r: 0.988, g: 0.988, b: 0.992 },  // White (slightly off-white for better contrast)
  { t: 0.75, r: 0.957, g: 0.647, b: 0.509 }, // Light red
  { t: 1.0, r: 0.706, g: 0.016, b: 0.149 }, // Deep red
];

function interpolateCoolwarm(t: number): { r: number; g: number; b: number } {
  const t_clamped = Math.max(0, Math.min(1, t));
  
  // Handle exact boundary cases to ensure pure colors at edges
  // This prevents any color mixing at the boundaries
  if (t_clamped <= 0) {
    const first = COOLWARM_KEYPOINTS[0];
    return { r: first.r, g: first.g, b: first.b };
  }
  if (t_clamped >= 1.0) {
    const last = COOLWARM_KEYPOINTS[COOLWARM_KEYPOINTS.length - 1];
    return { r: last.r, g: last.g, b: last.b };
  }
  
  // Find the two keypoints to interpolate between
  // We need to find the interval [keypoints[i].t, keypoints[i+1].t] that contains t_clamped
  for (let i = 0; i < COOLWARM_KEYPOINTS.length - 1; i++) {
    const p0 = COOLWARM_KEYPOINTS[i];
    const p1 = COOLWARM_KEYPOINTS[i + 1];
    
    // Check if t_clamped is in this interval
    // Use <= for the upper bound to handle the case where t_clamped equals p1.t (except for the last interval)
    if (t_clamped >= p0.t && (i === COOLWARM_KEYPOINTS.length - 2 ? t_clamped <= p1.t : t_clamped < p1.t)) {
      // Linear interpolation between keypoints
      const range = p1.t - p0.t;
      const u = range > 0 ? (t_clamped - p0.t) / range : 0;
      
      return {
        r: p0.r + (p1.r - p0.r) * u,
        g: p0.g + (p1.g - p0.g) * u,
        b: p0.b + (p1.b - p0.b) * u,
      };
    }
  }
  
  // Fallback (should never reach here, but just in case)
  const last = COOLWARM_KEYPOINTS[COOLWARM_KEYPOINTS.length - 1];
  return { r: last.r, g: last.g, b: last.b };
}

export function getViridisColor(t: number): { r: number; g: number; b: number } {
  const color = interpolateCoolwarm(t);
  return {
    r: Math.round(255 * Math.max(0, Math.min(1, color.r))),
    g: Math.round(255 * Math.max(0, Math.min(1, color.g))),
    b: Math.round(255 * Math.max(0, Math.min(1, color.b))),
  };
}

/**
 * Get Coolwarm color as normalized RGB (0-1 range) for Three.js
 */
export function getViridisColorNormalized(t: number): { r: number; g: number; b: number } {
  const color = interpolateCoolwarm(t);
  return {
    r: Math.max(0, Math.min(1, color.r)),
    g: Math.max(0, Math.min(1, color.g)),
    b: Math.max(0, Math.min(1, color.b)),
  };
}

