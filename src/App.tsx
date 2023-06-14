/*
* 2023-06-15 后记
*
* 看网上的教程以及 GPT4 的建议, 选用了 TypeScript + React Hook + Three.js 的组合方案.
* 但写的过程中发现这很乱, 因为 Three 有自己的一套状态维护, 比如 geometry 中的顶点坐标的更新, 而这些对 React 是不可见的.
* 而顶点的更新那里, 又弄了一个 ForceUpdate 的 Recoil State 才解决, 也不知道具体原理.
* 总之最后用了很多 tricks, 终于让程序框架丑陋地跑起来了.
*
* 而这还只是噩梦的开始, 等到写数值求解的部分才发现, JS 的数值计算生态更是离谱.
* 就矩阵而言, 有用 number[][] 的, 还有 Three 的 Matrix3, 有 linear-algebra-js 的 DenseMatrix 等等,
* 就没有一种像 NumPy 一样的统一一点的生态.
* 另外语言语法自由度的限制也导致没法像 Python 那样 `A-B`,
* 而是必须什么 `A.clone().subtract(B)`, 对, 你还得搞清楚这个 operator 是不是 inplace 的,
* 然后加个 .clone() 如果需要. 比如 Three 的向量加减法就默认 inplace, 稍不留神就导致奇怪的 bug.
*
* 另外, 援引的这两个数值库都很老, 也没有 npm 的版本, 于是没法 vite build 进来,
* 只能 build 之后手动复制到 dist 文件夹中. 当然, 可能有什么更好的方案, 但我并不知道.
* 总之呢, 如果有下一次尝试, 可能会改用 Cpp + Eigen + imGUI + OpenGL 了, 虽然也没用过就是了.
*
*/

import { useEffect, useRef, useState } from 'react';
import { useDrag } from 'react-use-gesture';

import { Box3, BufferAttribute, BufferGeometry, Matrix3, Mesh, MeshPhongMaterial, PerspectiveCamera, Vector3 } from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { Sky } from "@react-three/drei";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import Select from 'react-select'

import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import { CamCfgAtom, ForceUpdateAtom, LightCfgAtom, VertexType, vertexTypeToColor } from './constants';
import { Matrix3x3Add, sphericalToCartesian } from "./utils";


// 控制摄像机旋转的组件
export const CamPosCtl = () => {
  const [cameraConfig, setCameraConfig] = useRecoilState(CamCfgAtom);
  const sensitivity = 100;
  const bounds = {top: -sensitivity, bottom: sensitivity};
  const bind = useDrag(({offset: [offsetX, offsetY]}) => {
    const theta = -offsetX / sensitivity * Math.PI / 2;
    const phi = (-offsetY * 0.9999 / sensitivity + 1) * Math.PI / 2;
    setCameraConfig({...cameraConfig, theta, phi});
  }, {bounds: bounds});
  return (
    <div {...bind()} className={'cam-pos-ctl'}>Camera</div>
  )
};


// 控制光源旋转的组件
export const LightPosCtl = () => {
  const [lightConfig, setLightConfig] = useRecoilState(LightCfgAtom);
  const sensitivity = 100;
  const bounds = {top: -sensitivity, bottom: sensitivity};
  const bind = useDrag(({offset: [offsetX, offsetY]}) => {
    const theta = offsetX / sensitivity * Math.PI / 2;
    const phi = (offsetY * 0.7 / sensitivity + 1) * Math.PI / 2;
    setLightConfig({...lightConfig, theta, phi});
  }, {bounds: bounds});
  return (
    <div {...bind()} className={'light-pos-ctl'}>Light</div>
  )
};


// 动点选择器开启按钮
export const DraggableSelectorOn = ({onClick}) => {
  return (
    <div className={'draggable-selector-on-button'} onClick={onClick}>2|Draggable</div>
  )
};


// 定点选择器开启按钮
export const FixedSelectorOn = ({onClick}) => {
  return (
    <div className={'fixed-selector-on-button'} onClick={onClick}>1|Fixed &nbsp;&nbsp;&nbsp;</div>
  )
};


// 取消选择器开启按钮
export const DeSelectorOn = ({onClick}) => {
  return (
    <div className={'de-selector-on-button'} onClick={onClick}>3|De-Select</div>
  )
};


// 矩形选择器
export const RectSelector = ({selectorType, onDragDone}) => {
  type RectType = {x: number, y: number, dx: number, dy: number};
  const [draggingRect, setDraggingRect] = useState<RectType>({x: 0, y: 0, dx: 0, dy: 0});

  const bind = useDrag(({initial: [x0, y0], xy: [x, y], first, last}) => {
    const x0_ = x0 / window.innerWidth * 2 - 1;
    const y0_ = -(y0 / window.innerHeight * 2 - 1);
    const x_ = x / window.innerWidth * 2 - 1;
    const y_ = -(y / window.innerHeight * 2 - 1);
    const xMin = Math.min(x0_, x_);
    const xMax = Math.max(x0_, x_);
    const yMin = Math.min(y0_, y_);
    const yMax = Math.max(y0_, y_);
    setDraggingRect({x: Math.min(x0, x), y: Math.min(y0, y), dx: Math.abs(x0 - x), dy: Math.abs(y0 - y)});
    if (first) {
      console.log("start dragging");
    } else if (last) {
      console.log("end dragging")
      onDragDone({xMin, xMax, yMin, yMax});
    }
  })

  let borderColor;
  if (selectorType === "drag") {borderColor = '#ff5f9f';}
  else if (selectorType === "fixed") {borderColor = '#cba417';}
  else {borderColor = '#3594ff';}

  let backgroundColor;
  if (selectorType === "drag") {backgroundColor = '#ff5f9f10';}
  else if (selectorType === "fixed") {backgroundColor = '#ffdf5f10';}
  else {backgroundColor = '#5f5fff10';}

  return (
    <>
      <div {...bind()} style={{
        top: 0, left: 0, width: '100%', height: '100%',
        border: `0.3rem solid ${borderColor}`,
        backgroundColor: backgroundColor,
        boxSizing: 'border-box',
        position: 'absolute',
        cursor: 'crosshair',
      }}/>
      <div style={{
        top: draggingRect.y, left: draggingRect.x,
        width: draggingRect.dx, height: draggingRect.dy,
        border: `1px solid ${borderColor}`,
        backgroundColor: backgroundColor,
        boxSizing: 'border-box',
        position: 'absolute',
      }}/>
  </>
  )
}


// 选出经过 camera 映射后, 会落入矩形 [xMin, xMax, yMin, yMax] 内的顶点
const getVerticesWithinRect = ({verticesPosRef, meshRef, cameraRef, xMin, xMax, yMin, yMax}): Set<number> => {
  // xMin, xMax, yMin, yMax ∈ [-1, 1]
  const vertices = new Set<number>();
  for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
    const pos3D = new Vector3(verticesPosRef.current[3 * idx], verticesPosRef.current[3 * idx + 1], verticesPosRef.current[3 * idx + 2]);
    const posScreen = pos3D.clone().applyMatrix4(meshRef.current.matrixWorld).project(cameraRef.current);   // 注: 如果模型没有缩放旋转平移, 则无需进行 matrixWorld 变换
    if (posScreen.x >= xMin && posScreen.x <= xMax && posScreen.y >= yMin && posScreen.y <= yMax) {
      vertices.add(idx);
    }
  }
  return vertices;
}


// 根据 faces 建立邻居表 (邻居表: 查询给定顶点 i 的全部邻居顶点 j)
type NeighborTableType = Map<number, Set<number>>;
const buildNeighborTable = (faces: Uint32Array): NeighborTableType => {
  const neighborTable = new Map();
  for (let faceIdx = 0; faceIdx < faces.length / 3; faceIdx++) {
    const [vA, vB, vC] = faces.slice(3 * faceIdx, 3 * faceIdx + 3);   // 该三角形三个顶点的编号
    if (!neighborTable.has(vA)) {neighborTable.set(vA, new Set());}
    if (!neighborTable.has(vB)) {neighborTable.set(vB, new Set());}
    if (!neighborTable.has(vC)) {neighborTable.set(vC, new Set());}
    neighborTable.get(vA).add(vB);
    neighborTable.get(vA).add(vC);
    neighborTable.get(vB).add(vA);
    neighborTable.get(vB).add(vC);
    neighborTable.get(vC).add(vA);
    neighborTable.get(vC).add(vB);
  }
  return neighborTable;
}


// 根据 faces 建立边对顶点表 (边对顶点表: 查询给定边 i->j 的对顶点 k, 形如 "123-456" -> 0)
type OppositeVtxIdOfEdgeType = Map<string, number>;
const buildOppositeVtxIdOfEdge = (faces: Uint32Array): OppositeVtxIdOfEdgeType => {
  const oppositeVtxIdOfEdge = new Map();
  for (let faceIdx = 0; faceIdx < faces.length / 3; faceIdx++) {
    const [vA, vB, vC] = faces.slice(3 * faceIdx, 3 * faceIdx + 3);   // 该三角形三个顶点的编号
    oppositeVtxIdOfEdge.set(`${vA}-${vB}`, vC);
    oppositeVtxIdOfEdge.set(`${vB}-${vC}`, vA);
    oppositeVtxIdOfEdge.set(`${vC}-${vA}`, vB);
  }
  return oppositeVtxIdOfEdge;
}


// 建立权重表 (查询给定边 i-j 的权重 w_ij, 形如 "123-137" -> 0.5)
type WIJType = Map<string, number>;
const buildWij = (verticesPos: Float32Array, faces: Uint32Array, neighborTable: NeighborTableType, oppositeVtxIdOfEdge: OppositeVtxIdOfEdgeType): WIJType => {
  const wIJ = new Map();
  for (let vtxIdx = 0; vtxIdx < verticesPos.length / 3; vtxIdx++) {
    const neighborVtxIds = Array.from(neighborTable.get(vtxIdx));
    for (const neighborVtxId of neighborVtxIds) {

      // 计算 ∠ikj 的 cot 值, 其中 i-j 为边, k 为 i-j 边对顶点
      const calcCot = (idxI, idxJ, idxOpposite) => {
        const vtxPosI = new Vector3(verticesPos[3 * idxI], verticesPos[3 * idxI + 1], verticesPos[3 * idxI + 2]);
        const vtxPosJ = new Vector3(verticesPos[3 * idxJ], verticesPos[3 * idxJ + 1], verticesPos[3 * idxJ + 2]);
        const vtxPosK = new Vector3(verticesPos[3 * idxOpposite], verticesPos[3 * idxOpposite + 1], verticesPos[3 * idxOpposite + 2]);
        const vecKI = new Vector3().subVectors(vtxPosK, vtxPosI).normalize();
        const vecKJ = new Vector3().subVectors(vtxPosK, vtxPosJ).normalize();
        const cosine = vecKI.dot(vecKJ);
        const sine = vecKI.clone().cross(vecKJ).length();
        let cotTheta = cosine / sine;
        cotTheta = Math.max(cotTheta, 0);   // 如果不约束成正数, 则在某些模型上就会有问题, 比如 person.obj 的帽子会崩坏..
        return cotTheta;
      }

      // alpha_ij
      let cotAlpha;
      const edgeKeyIJ = `${vtxIdx}-${neighborVtxId}`;
      const oppositeVtxIdOfEdgeIJ = oppositeVtxIdOfEdge.get(edgeKeyIJ);
      if (oppositeVtxIdOfEdgeIJ !== undefined) { // 若该边是边界边, 则没有对顶点
        cotAlpha = calcCot(vtxIdx, neighborVtxId, oppositeVtxIdOfEdgeIJ);
      }

      // beta_ij
      let cotBeta;
      const edgeKeyJI = `${neighborVtxId}-${vtxIdx}`;
      const oppositeVtxIdOfEdgeJI = oppositeVtxIdOfEdge.get(edgeKeyJI);
      if (oppositeVtxIdOfEdgeJI !== undefined) { // 若该边是边界边, 则没有对顶点
        cotBeta = calcCot(vtxIdx, neighborVtxId, oppositeVtxIdOfEdgeJI);
      }

      // w_ij
      let w;
      if (cotAlpha === undefined) {w = cotBeta;}
      else if (cotBeta === undefined) {w = cotAlpha;}
      else {w = 0.5 * (cotAlpha + cotBeta);}
      wIJ.set(edgeKeyIJ, w);    // i->j
      wIJ.set(edgeKeyJI, w);    // j->i
    }
  }

  return wIJ;
}


// 根据 wIJ 构建稀疏系数矩阵 L
type MatrixLType = Array<Array<[number, number]>>;    // L[i] 是一个数组, 存储了第 i 行, 每个元素是 [j, w_ij], 每个行数组都是无序的
const buildMatrixL = (wIJ: WIJType, nrVertices: number, neighborTable: NeighborTableType): MatrixLType => {
  const matrixL = [];
  for (let i = 0; i < nrVertices; i++) {
    matrixL[i] = matrixL[i] || [];
    let L_ii = 0;
    for (const j of neighborTable.get(i)) {
      const w_ij = wIJ.get(`${i}-${j}`);
      const L_ij = -w_ij;
      matrixL[i].push([j, L_ij]); // L_ij = -w_ij
      L_ii += w_ij;               // L_ii += w_ij
    }
    matrixL[i].push([i, L_ii]);   // L_ii = Σw_ij
  }
  return matrixL;
}


// 计算旋转矩阵 Ri
type MatrixRType = number[3][3];
const calcRotationMatrices = (verticesOriginalPos, verticesPos, neighborTable, wIJ): MatrixRType[] => {
  const p = verticesOriginalPos;
  const pPrime = verticesPos;
  const nrVertices = p.length / 3;
  const R = [];

  for (let i = 0; i < nrVertices; i++) {
    // Si = Σj(w_ij * (p_j - p_i) * (p'_j - p'_i)^T)
    const Si = [0, 0, 0,  0, 0, 0,  0, 0, 0];
    for (const j of neighborTable.get(i)) {
      const w_ij = wIJ.get(`${i}-${j}`) as number;
      const p_i = new Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      const p_j = new Vector3(p[3 * j], p[3 * j + 1], p[3 * j + 2]);
      const pPrime_i = new Vector3(pPrime[3 * i], pPrime[3 * i + 1], pPrime[3 * i + 2]);
      const pPrime_j = new Vector3(pPrime[3 * j], pPrime[3 * j + 1], pPrime[3 * j + 2]);
      const eij = new Vector3().subVectors(p_j, p_i);
      const eijPrime = new Vector3().subVectors(pPrime_j, pPrime_i);
      const outer = [eij.x * eijPrime.x, eij.x * eijPrime.y, eij.x * eijPrime.z,    // (p_j - p_i) * (p'_j - p'_i)^T
                     eij.y * eijPrime.x, eij.y * eijPrime.y, eij.y * eijPrime.z,
                     eij.z * eijPrime.x, eij.z * eijPrime.y, eij.z * eijPrime.z]
      outer.forEach((value, index) => {Si[index] += w_ij * value;});                // Si += w_ij * (p_j - p_i) * (p'_j - p'_i)^T
    }

    // 最优旋转: V * U^T
    const SiMat = [Si.slice(0, 3), Si.slice(3, 6), Si.slice(6, 9)];
    const USVT = numeric.svd(SiMat);
    if (numeric.det(USVT.U) * numeric.det(USVT.V) < 0) {  // 如果 Ri 不是旋转矩阵 (行列式不为 +1), 则将第三列反向, 以尽量小的代价将其变为旋转矩阵
      USVT.U[0][2] *= -1; USVT.U[1][2] *= -1; USVT.U[2][2] *= -1;
    }
    const Ri = numeric.dot(USVT.V, numeric.transpose(USVT.U));  // Ri := V * U^T
    R.push(Ri);
  }

  return R;
}


// 显示模型的组件
export const TriMesh = ({modelLoaded, verticesPosRef, verticesTypeRef, cameraRef, geometryRef, meshRef}) => {

  // 仅仅是挂个 hook, 用于触发染色顶点的更新 (否则大概就得维护一个列表, 逐个更新每个 sphere)
  const forceUpdate = useRecoilValue(ForceUpdateAtom);

  // 设置相机位姿
  const { camera }: { camera: PerspectiveCamera } = useThree();
  const cameraConfig = useRecoilValue(CamCfgAtom);
  cameraRef.current = camera;   // 回传给上一层, 计算矩形框选时用于反投影
  useEffect(() => {
    const {theta, phi, distance} = cameraConfig;
    const cameraPos = sphericalToCartesian({theta, phi, distance});
    camera.position.set(...cameraPos);
    camera.lookAt(0, 0, 0);
  })

  return (
    <>
      {/* 模型主体 mesh */}
      {modelLoaded && (
        <mesh
          ref={meshRef}
          geometry={geometryRef.current}
          material={new MeshPhongMaterial({color: 0x049ef4})}
        />
      )}

      {/* 染色的顶点 */}
      {modelLoaded && (
        verticesTypeRef.current
          .map((vtxType, vtxId) => [vtxType, vtxId])
          .filter(([vtxType, vtxId]) => vtxType !== VertexType.Calculated)
          .map(([vtxType, vtxId]) => (
          <mesh key={vtxId} position={[...verticesPosRef.current.slice(3 * vtxId, 3 * vtxId + 3)]}>
            <sphereGeometry args={[vtxType === VertexType.Calculated ? 0.002 : 0.004, 5, 5]}/>
            <meshPhongMaterial color={vertexTypeToColor(vtxType)}/>
          </mesh>
        ))
      )}
    </>
  )
}


export const App = () => {

  // 模型
  const meshRef = useRef<Mesh>();
  const geometryRef = useRef<BufferGeometry>();
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);

  // 相机参数
  const cameraRef = useRef();

  // 光源方向
  const [lightConfig, ] = useRecoilState(LightCfgAtom);
  const lightPos = sphericalToCartesian({theta: lightConfig.theta, phi: lightConfig.phi, distance: lightConfig.distance});

  // 顶点选择器类型: "none" | "drag" | "fixed" | "calc"
  const [selectorType, setSelectorType] = useState("none");

  // 所有顶点的 [变形前] 坐标
  const verticesOriginalPosRef = useRef<Float32Array>();

  // 所有顶点的 [当前(变形后)] 坐标
  const verticesPosRef = useRef<Float32Array>();

  // 每个顶点的状态: fixed | draggable | calculated
  const verticesTypeRef = useRef<VertexType[]>();

  // ARAP 算法中, 避免重复计算用
  const neighborTableRef = useRef<NeighborTableType>();         // 顶点邻接表
  const oppositeVtxIdOfEdgeRef = useRef<Map<string, number>>(); // 有向边 i->j 所在三角形的 ｢对点｣ 编号. 对于每个模型只需要构造一次
  const wIJRef = useRef<Map<string, number>>();                 // w_ij: 边 i-j 的权重, 对于每个模型只需要计算一次
  const LMatrixRef = useRef<MatrixLType>();                     // L: 系数矩阵, 对于每个模型只需要计算一次, 之后只需要取所需的 n 行 n 列即可;

  // 加载模型, 初始化各数据
  const modelOptions = [
    {label: "cactus", value: "cactus.obj"},
    {label: "person", value: "person.obj"},
  ]
  const loadModel = (modelName: "string") => {
    console.log("loading model: ", modelName)
    // 加载 obj 模型, 合并顶点, 转为 vertices 和 faces 数组
    const loader = new OBJLoader();
    loader.load(modelName, (object) => {
      object.traverse((child) => {
        if (!(child instanceof Mesh)) {
          return;
        }
        // 用 mergeVertices() 合并相同的顶点 (不知道为啥读取的模型是 non-indexed 的)
        child.geometry.deleteAttribute('normal');   // 如果不删除, 就没法 merge
        child.geometry.deleteAttribute('uv');
        child.geometry = mergeVertices(child.geometry);
        const vertices = child.geometry.attributes.position.array as Float32Array;
        const faces = (child.geometry.index as BufferAttribute).array as Uint32Array;

        // 初始化每个顶点的类型 (VertexType)
        verticesTypeRef.current = new Array(vertices.length / 3).fill(VertexType.Calculated);

        // 构建 geometry 数据结构
        const geom = new BufferGeometry();
        geom.setAttribute("position", new BufferAttribute(vertices, 3));
        geom.setIndex(new BufferAttribute(faces, 1));
        geom.computeVertexNormals();

        // 将模型居中, 使得 bbox 的中心点为 [0,0,0]. (注: 通过修改 verticesPosRef.current 实现)
        geom.center();

        // 将模型的 bbox 缩放到 [-0.5, 0.5] 立方体内. (注: 通过修改 verticesPosRef.current 实现)
        const bboxSize = (new Box3()).setFromBufferAttribute(geom.attributes.position as BufferAttribute).getSize(new Vector3());
        const maxSize = Math.max(bboxSize.x, bboxSize.y, bboxSize.z);
        const scale = 1 / maxSize;
        geom.scale(scale, scale, scale);

        // 仅与模型本身有关(绑定)的信息
        verticesPosRef.current = vertices;
        verticesOriginalPosRef.current = vertices.slice();
        geometryRef.current = geom;
        neighborTableRef.current = buildNeighborTable(faces);
        oppositeVtxIdOfEdgeRef.current = buildOppositeVtxIdOfEdge(faces);
        wIJRef.current = buildWij(vertices, faces, neighborTableRef.current, oppositeVtxIdOfEdgeRef.current);
        LMatrixRef.current = buildMatrixL(wIJRef.current, verticesPosRef.current.length / 3, neighborTableRef.current);

        // 设置模型已加载的状态, 从而允许 React 渲染 Mesh
        setModelLoaded(true);
        setForceUpdate(Math.random());
        setCanVtxPosUpdate(false);
      });
    });
  }
  useEffect(() => {
    loadModel(modelOptions[0].value);
  }, []);

  // 基于 ARAP 算法, 计算并更新待计算的顶点坐标
  const updateVtxPosARAP = () => {

    // 统计哪些顶点的坐标是待计算的
    const verticesToCalc = verticesTypeRef.current.map((vtxType, vtxId) => vtxType === VertexType.Calculated ? vtxId : -1).filter(vtxId => vtxId !== -1);

    // 构建线性方程 Lx = b, 解算这些待计算顶点的坐标
    const newToOri = verticesToCalc;
    const oriToNew = new Array<number>(verticesPosRef.current.length).fill(-1);
    newToOri.forEach((oriId, newId) => oriToNew[oriId] = newId);

    // 系数向量 b
    const R = calcRotationMatrices(verticesOriginalPosRef.current, verticesPosRef.current, neighborTableRef.current, wIJRef.current);
    const b = DenseMatrix.zeros(verticesToCalc.length, 3);
    for (let newI = 0; newI < verticesToCalc.length; newI++) {
      // bi := Σj 0.5 * (Ri + Rj) * (pi - pj)
      const oriI = newToOri[newI];
      const Ri = R[oriI];
      const pi = new Vector3(...verticesOriginalPosRef.current.slice(oriI * 3, oriI * 3 + 3));
      const neighborVtxIds = Array.from(neighborTableRef.current.get(oriI));
      let [bix, biy, biz] = [0, 0, 0]
      for (const oriJ of neighborVtxIds) {
        const newJ = oriToNew[oriJ];
        if (newJ !== -1) {
          const Rj = R[oriJ];
          const RiPlusRj = Matrix3x3Add(Ri, Rj);

          // // fixme: Debug: 退化为 Laplacian 矩阵 (后续可以加一个按钮来操控)
          // [RiPlusRj[0][0], RiPlusRj[0][1], RiPlusRj[0][2]] = [2, 0, 0];
          // [RiPlusRj[1][0], RiPlusRj[1][1], RiPlusRj[1][2]] = [0, 2, 0];
          // [RiPlusRj[2][0], RiPlusRj[2][1], RiPlusRj[2][2]] = [0, 0, 2];

          const RiPlusRjMat = new Matrix3();
          RiPlusRjMat.set(...RiPlusRj[0], ...RiPlusRj[1], ...RiPlusRj[2]);  // 行优先 (https://threejs.org/docs/?q=matrix#api/zh/math/Matrix4)
          const pj = new Vector3(...verticesOriginalPosRef.current.slice(oriJ * 3, oriJ * 3 + 3));
          const w_ij = wIJRef.current.get(`${oriI}-${oriJ}`);
          const bi = (pi.clone().sub(pj)).multiplyScalar(0.5 * w_ij).applyMatrix3(RiPlusRjMat);
          bix += bi.x;
          biy += bi.y;
          biz += bi.z;
        }
      }
      b.set(bix, newI, 0);
      b.set(biy, newI, 1);
      b.set(biz, newI, 2);
    }

    // 系数矩阵 L
    const oriL = LMatrixRef.current as MatrixLType;
    const pPrime = verticesPosRef.current as Float32Array;
    const triplet = new Triplet(verticesToCalc.length, verticesToCalc.length);
    for (let newI = 0; newI < verticesToCalc.length; newI++) {
      const oriI = newToOri[newI];
      for (const [oriJ, oriLij] of oriL[oriI]) {
        const newJ = oriToNew[oriJ];
        if (newJ !== -1) {    // j 是待计算的顶点, 则加入系数矩阵
          triplet.addEntry(oriLij, newI, newJ);
        } else {    // j 是已知的顶点, 则不加入系数矩阵, 而是加入 b
          b.set(b.get(newI, 0) - oriLij * pPrime[oriJ * 3 + 0], newI, 0);   // 注意: 这里是 -(-wij), 所以是 -oriLij!!!
          b.set(b.get(newI, 1) - oriLij * pPrime[oriJ * 3 + 1], newI, 1);
          b.set(b.get(newI, 2) - oriLij * pPrime[oriJ * 3 + 2], newI, 2);
        }
      }
    }
    const L = SparseMatrix.fromTriplet(triplet);

    // 求解 Lp' = b
    const A = L;
    const bx = b.subMatrix(0, verticesToCalc.length, 0, 1);
    const by = b.subMatrix(0, verticesToCalc.length, 1, 2);
    const bz = b.subMatrix(0, verticesToCalc.length, 2, 3);
    const llt = A.chol();
    const px = llt.solvePositiveDefinite(bx);
    const py = llt.solvePositiveDefinite(by);
    const pz = llt.solvePositiveDefinite(bz);

    // 将计算得到的坐标写入 verticesPosRef
    for (let i = 0; i < verticesToCalc.length; i++) {
      const oriI = newToOri[i];
      verticesPosRef.current[oriI * 3 + 0] = px.get(i, 0);
      verticesPosRef.current[oriI * 3 + 1] = py.get(i, 0);
      verticesPosRef.current[oriI * 3 + 2] = pz.get(i, 0);
    }

    // 释放内存, 否则会内存泄漏, 不一会就炸了 (要在 chrome task manager 看, 因为这个不是 JS Heap 内存, 而是 C++ asm.js 内存)
    memoryManager.deleteExcept([]);

    // 通知 three 更新模型
    (geometryRef.current as BufferGeometry).attributes.position.needsUpdate = true;
    setForceUpdate(Math.random());  // 触发 [染色顶点] 的重渲染 (但我不知道为啥这样就能解决..)
  }

  // 处理顶点拖拽
  const cameraConfig = useRecoilValue(CamCfgAtom);
  const setForceUpdate = useSetRecoilState(ForceUpdateAtom)
  const bind = useDrag(({delta: [dRight, dDown]}) => {
    const sensitivity = 100;
    const {theta, phi} = cameraConfig;
    const [xC, yC, zC] = sphericalToCartesian({theta, phi, distance: 1});
    const [xH, yH, zH] = [xC, 0, zC];
    const pointOrigin = new Vector3(0, 0, 0);
    const pointCamera = new Vector3(xC, yC, zC);
    const pointH = new Vector3(xH, yH, zH);
    const vecOC = pointCamera.clone().sub(pointOrigin).normalize();
    const vecOH = pointH.clone().sub(pointOrigin).normalize();
    const vecRight = vecOC.clone().cross(vecOH).normalize();
    const vecUp = vecOC.clone().cross(vecRight).normalize();
    if (pointCamera.y < 0) {  // 否则摄像机在 XZ 平面之下时的拖拽方向是错的, 上面的叉乘是在 y 正半轴画图推导的
      vecUp.multiplyScalar(-1);
      vecRight.multiplyScalar(-1);
    }
    const vecDelta = vecRight.clone().multiplyScalar(dRight).add(vecUp.clone().multiplyScalar(-dDown));
    // 更新拖拽指定的动点坐标
    for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
      if (verticesTypeRef.current[idx] === VertexType.Draggable) {
        verticesPosRef.current[3 * idx + 0] += vecDelta.x / sensitivity;
        verticesPosRef.current[3 * idx + 1] += vecDelta.y / sensitivity;
        verticesPosRef.current[3 * idx + 2] += vecDelta.z / sensitivity;
      }
    }
    // 通知 three 更新模型
    (geometryRef.current as BufferGeometry).attributes.position.needsUpdate = true;
    setForceUpdate(Math.random());  // 触发 [染色顶点] 的重渲染 (但我不知道为啥这样就能解决..)
  })

  // 持续不断地基于 ARAP 计算新坐标
  const [canVtxPosUpdate, setCanVtxPosUpdate] = useState(false);
  const timerRef = useRef<number>(0);
  useEffect(() => {
    if (canVtxPosUpdate && !timerRef.current) {   // 目前应该更新, 但还没上计时器, 就上计时器
      timerRef.current = setInterval(updateVtxPosARAP, 1);
    }
    if (!canVtxPosUpdate && timerRef.current) {   // 目前不应该更新, 但还没清除计时器, 就清除计时器
      clearInterval(timerRef.current);
      timerRef.current = 0;
    }
    return () => {  // 组件卸载时清除定时器
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = 0;
      }
    }
  }, [canVtxPosUpdate]);

  // 顶点选择器的快捷键: 1/2/3
  useEffect(() => {
    const handleKeyDown = (e) => {
      console.log(e)
      if (e.key === '1') {
        setSelectorType(selectorType === "fixed" ? "none": "fixed")
      } else if (e.key === '2') {
        setSelectorType(selectorType === "drag" ? "none": "drag")
      } else if (e.key === '3') {
        setSelectorType(selectorType === "calc" ? "none": "calc")
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectorType]);

  return (<>

    {/* Three-js */}
    <Canvas {...bind()}>
      {/* 天空和光源 */}
      <group>
        <Sky/>
        <ambientLight intensity={0.1}/>
        <pointLight position={lightPos} intensity={0.7}/>
        <pointLight position={[-13, 100, 3]} intensity={0.2}/>
        <pointLight position={[4, -100, 11]} intensity={0.2}/>
        <pointLight position={[100, 1, -7]} intensity={0.2}/>
        <pointLight position={[-100, -3, 9]} intensity={0.2}/>
        <pointLight position={[11, 4, -100]} intensity={0.2}/>
        <pointLight position={[-12, 14, 100]} intensity={0.2}/>
      </group>
      {/* 模型 */}
      <TriMesh
        modelLoaded={modelLoaded}
        verticesPosRef={verticesPosRef}
        verticesTypeRef={verticesTypeRef}
        cameraRef={cameraRef}
        geometryRef={geometryRef}
        meshRef={meshRef}
      />
    </Canvas>

    {/* 模型选择器 */}
    <span className="abs-text" style={{left: "5rem", bottom: "21.8rem"}}> Select a model:</span>
    <div className="model-selector">
      <Select
        options={modelOptions}
        defaultValue={modelOptions[0]}
        onChange={({label, value}) => loadModel(value)}
      />
    </div>


    {/* 相机和光源控件 */}
    <>
      <span className="abs-text" style={{left: "5rem", bottom: "16.4rem"}}> Drag to rotate:</span>
      <CamPosCtl/>
      <LightPosCtl/>
    </>

    {/* 顶点交互框选器 */}
    <div className="selector-container">
      <span style={{fontFamily: "sans-serif"}}>Vertex Selectors:</span>
      <FixedSelectorOn onClick={()=>{setSelectorType("fixed")}}/>
      <DraggableSelectorOn onClick={()=>{setSelectorType("drag")}}/>
      <DeSelectorOn onClick={()=>{setSelectorType("calc")}}/>
    </div>
    {selectorType !== "none" &&
      <RectSelector
        selectorType={selectorType}
        onDragDone={({xMin, xMax, yMin, yMax}) => {
          // 关闭矩形选择器
          setSelectorType("none");
          // 找出所有落入该矩形的顶点编号, // 将这些顶点设置为 Fixed 或 Draggable 或 Calc
          const verticesWithinRect = getVerticesWithinRect({verticesPosRef, meshRef, cameraRef, xMin, xMax, yMin, yMax});
          const types = verticesTypeRef.current as VertexType[];
          for (const idx of verticesWithinRect) {
            if (selectorType === "drag") {types[idx] = VertexType.Draggable;}
            else if (selectorType === "fixed") {types[idx] = VertexType.Fixed;}
            else if (selectorType === "calc") {types[idx] = VertexType.Calculated;}
          }
          // 记 fixed 和 draggable 均为 handle, 若没有 handle, 则不能更新顶点坐标, 因为没有绝对位置的约束了!
          const nrHandle = types.filter(type => type !== VertexType.Calculated).length;
          if (nrHandle > 0 && !canVtxPosUpdate) {setCanVtxPosUpdate(true);}
          if (nrHandle === 0 && canVtxPosUpdate) {setCanVtxPosUpdate(false);}
        }}
      />
    }

  </>)
}

export default App;
