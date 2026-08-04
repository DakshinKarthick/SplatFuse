export function bitonicStages(length) {
  if (length < 1 || (length & (length - 1)) !== 0) {
    throw new RangeError('bitonic sort length must be a positive power of two')
  }
  const stages = []
  for (let k = 2; k <= length; k *= 2) {
    for (let j = k / 2; j >= 1; j /= 2) stages.push({ j, k, length })
  }
  return stages
}

export function compareKeyPair(a, b) {
  return a[1] === b[1] ? a[0] - b[0] : a[1] - b[1]
}
