/**
 * Scaling report statistics and conservative comparisons of measured timing intervals.
 */

export function quantile(values, probability) {
  if (!values.length) throw new Error('Cannot summarize empty samples');
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * (position - lower);
}

export function summarize(values) {
  const median = quantile(values, 0.5);
  const lowerQuartile = quantile(values, 0.25);
  const upperQuartile = quantile(values, 0.75);
  return { median, lowerQuartile, upperQuartile, iqr: upperQuartile - lowerQuartile,
    minimum: Math.min(...values), maximum: Math.max(...values), count: values.length };
}

export function compareTiming(candidate, reference) {
  if (!candidate || !reference) return { classification: 'unavailable' };
  return { ratio: candidate.median / reference.median,
    classification: candidate.lowerQuartile > reference.upperQuartile ? 'slower' :
      candidate.upperQuartile < reference.lowerQuartile ? 'faster' : 'overlapping' };
}
