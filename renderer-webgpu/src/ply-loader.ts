export interface SplatData {
  vertexCount: number;
  positions: Float32Array; // x, y, z
  scales: Float32Array;    // scale_x, scale_y, scale_z
  rotations: Float32Array; // q_w, q_x, q_y, q_z
  colors: Float32Array;    // r, g, b (from f_dc)
  opacities: Float32Array; // opacity
}

export async function loadPLY(url: string): Promise<SplatData> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  
  // Basic PLY parser
  const textDecoder = new TextDecoder('utf-8');
  let headerText = '';
  let headerEnd = 0;
  
  for (let i = 0; i < buffer.byteLength; i++) {
    const view = new Uint8Array(buffer, i, 10);
    const str = textDecoder.decode(view);
    if (str.startsWith('end_header')) {
      headerEnd = i + 11;
      headerText = textDecoder.decode(new Uint8Array(buffer, 0, headerEnd));
      break;
    }
  }

  if (!headerText) {
    throw new Error('Invalid PLY file');
  }

  const lines = headerText.split('\n');
  let vertexCount = 0;
  let isBinary = false;

  for (const line of lines) {
    if (line.startsWith('element vertex')) {
      vertexCount = parseInt(line.split(' ')[2]);
    }
    if (line.startsWith('format binary_little_endian')) {
      isBinary = true;
    }
  }

  if (!isBinary) {
    throw new Error('Only binary_little_endian PLY files are supported');
  }

  // Find property offsets
  let currentOffset = 0;
  const properties: { name: string, offset: number, type: string }[] = [];
  let inVertexElement = false;

  for (const line of lines) {
    const parts = line.trim().split(' ');
    if (parts[0] === 'element') {
      inVertexElement = parts[1] === 'vertex';
    } else if (parts[0] === 'property' && inVertexElement) {
      const type = parts[1];
      const name = parts[2];
      properties.push({ name, offset: currentOffset, type });
      if (type === 'float' || type === 'float32') currentOffset += 4;
      else if (type === 'uchar' || type === 'uint8') currentOffset += 1;
      else if (type === 'int') currentOffset += 4;
      else throw new Error(`Unsupported property type: ${type}`);
    }
  }

  const vertexSize = currentOffset;
  const dataView = new DataView(buffer, headerEnd);

  const positions = new Float32Array(vertexCount * 3);
  const scales = new Float32Array(vertexCount * 3);
  const rotations = new Float32Array(vertexCount * 4);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);

  // Map properties to indices
  const getProp = (name: string) => properties.find(p => p.name === name);
  const pX = getProp('x'), pY = getProp('y'), pZ = getProp('z');
  const pScaleX = getProp('scale_0'), pScaleY = getProp('scale_1'), pScaleZ = getProp('scale_2');
  const pRot0 = getProp('rot_0'), pRot1 = getProp('rot_1'), pRot2 = getProp('rot_2'), pRot3 = getProp('rot_3');
  const pOpac = getProp('opacity');
  const pFdc0 = getProp('f_dc_0'), pFdc1 = getProp('f_dc_1'), pFdc2 = getProp('f_dc_2');

  const SH_C0 = 0.28209479177387814;

  for (let i = 0; i < vertexCount; i++) {
    const base = i * vertexSize;

    if (pX && pY && pZ) {
      positions[i * 3 + 0] = dataView.getFloat32(base + pX.offset, true);
      positions[i * 3 + 1] = dataView.getFloat32(base + pY.offset, true);
      positions[i * 3 + 2] = dataView.getFloat32(base + pZ.offset, true);
    }

    if (pScaleX && pScaleY && pScaleZ) {
      scales[i * 3 + 0] = Math.exp(dataView.getFloat32(base + pScaleX.offset, true));
      scales[i * 3 + 1] = Math.exp(dataView.getFloat32(base + pScaleY.offset, true));
      scales[i * 3 + 2] = Math.exp(dataView.getFloat32(base + pScaleZ.offset, true));
    }

    if (pRot0 && pRot1 && pRot2 && pRot3) {
      let rw = dataView.getFloat32(base + pRot0.offset, true);
      let rx = dataView.getFloat32(base + pRot1.offset, true);
      let ry = dataView.getFloat32(base + pRot2.offset, true);
      let rz = dataView.getFloat32(base + pRot3.offset, true);
      
      const len = Math.sqrt(rw*rw + rx*rx + ry*ry + rz*rz);
      rotations[i * 4 + 0] = rw / len;
      rotations[i * 4 + 1] = rx / len;
      rotations[i * 4 + 2] = ry / len;
      rotations[i * 4 + 3] = rz / len;
    }

    if (pOpac) {
      const op = dataView.getFloat32(base + pOpac.offset, true);
      opacities[i] = 1.0 / (1.0 + Math.exp(-op)); // Sigmoid
    }

    if (pFdc0 && pFdc1 && pFdc2) {
      const r = dataView.getFloat32(base + pFdc0.offset, true);
      const g = dataView.getFloat32(base + pFdc1.offset, true);
      const b = dataView.getFloat32(base + pFdc2.offset, true);
      colors[i * 3 + 0] = Math.max(0, Math.min(1, 0.5 + SH_C0 * r));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * g));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * b));
    }
  }

  return {
    vertexCount,
    positions,
    scales,
    rotations,
    colors,
    opacities
  };
}
