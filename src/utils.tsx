// 球坐标系 --> 直角坐标系
export const sphericalToCartesian = ({theta, phi, distance}): [number, number, number] => {
  const x = distance * Math.sin(phi) * Math.sin(theta);
  const y = distance * Math.cos(phi);
  const z = distance * Math.sin(phi) * Math.cos(theta);
  return [x, y, z];
}


// 3x3 矩阵加法
export const Matrix3x3Add = (A: number[3][3], B: number[3][3]) => {
  const C = new Array<number[3]>(3);
  for (let i = 0; i < 3; i++) {
    C[i] = new Array<number>(3);
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return C;
}


// export const parseObjFile = (content: string): [Float32Array, Uint32Array] => {
//   const lines = content.split("\n");
//   const vertices: number[] = [];  // [x1,y1,z1,  x2,y2,z2,  x3,y3,z3, ...]
//   const indices: number[] = [];   // [i1,i2,i3,  i4,i5,i6,  i7,i8,i9, ...]
//   for (let i = 0; i < lines.length; i++) {
//     const line: string[] = lines[i].trim().split(" ");
//     if (line[0] === "v") {
//       vertices.push(parseFloat(line[1]), parseFloat(line[2]), parseFloat(line[3]));
//     } else if (line[0] === "f") {
//       indices.push(parseInt(line[1]) - 1, parseInt(line[2]) - 1, parseInt(line[3]) - 1);
//     }
//   }
//   return [Float32Array.from(vertices), Uint32Array.from(indices)];
// };


// const reader = new FileReader();
// reader.onload = () => {
//   const content = reader.result as string;
//   const [vertices, faces] = parseObjFile(content);
// };
// reader.readAsText(this.files[0]);


// // 键盘事件监听器
// export const KeyboardListener = () => {
//   useEffect(() => {
//     const handleKeyDown = (e) => {
//       console.log(e)
//     }
//     document.addEventListener('keydown', handleKeyDown);
//     return () => {
//       document.removeEventListener('keydown', handleKeyDown);
//     };
//   }, []);
//   return <></>;
// };
