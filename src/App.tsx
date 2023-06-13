import { useEffect, useRef, useState } from 'react';
import { useDrag } from 'react-use-gesture';


import { Box3, BufferAttribute, BufferGeometry, Mesh, MeshPhongMaterial, PerspectiveCamera, Vector3 } from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { Sky } from "@react-three/drei";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";

import { useRecoilState } from 'recoil';
import { CamCfgAtom, ForceUpdateAtom, LightCfgAtom, VertexType } from './atoms';
import { sphericalToCartesian } from "./utils";


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
    <div {...bind()} className={'cam-pos-ctl'}>Cam</div>
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
    <div {...bind()} className={'light-pos-ctl'}>Lit</div>
  )
};


// 动点选择器开启组件
export const DraggableSelectorOn = ({onClick}) => {
  return (
    <div className={'draggable-selector-on-button'} onClick={onClick}>Drag Select</div>
  )
};


// 定点选择器开启组件
export const FixedSelectorOn = ({onClick}) => {
  return (
    <div className={'fixed-selector-on-button'} onClick={onClick}>Fixed Select</div>
  )
};


// 取消选择器开启组件
export const DeSelectorOn = ({onClick}) => {
  return (
    <div className={'de-selector-on-button'} onClick={onClick}>De Select</div>
  )
};




// 矩形顶点选择器
export const RectSelector = ({selectorType, onDragDone}) => {
  const bind = useDrag(({initial: [x0, y0], xy: [x, y], first, last}) => {
    const x0_ = x0 / window.innerWidth * 2 - 1;
    const y0_ = -(y0 / window.innerHeight * 2 - 1);
    const x_ = x / window.innerWidth * 2 - 1;
    const y_ = -(y / window.innerHeight * 2 - 1);
    const xMin = Math.min(x0_, x_);
    const xMax = Math.max(x0_, x_);
    const yMin = Math.min(y0_, y_);
    const yMax = Math.max(y0_, y_);
    if (first) {
      console.log("start dragging");
    } else if (last) {
      console.log("end dragging")
      onDragDone({xMin, xMax, yMin, yMax});
    }
  })

  let borderColor;
  if (selectorType === "drag") {borderColor = '#ff5f9f';}
  else if (selectorType === "fixed") {borderColor = '#ffdf5f';}
  else {borderColor = '#5f5fff';}

  let backgroundColor;
  if (selectorType === "drag") {backgroundColor = '#ff5f9f10';}
  else if (selectorType === "fixed") {backgroundColor = '#ffdf5f10';}
  else {backgroundColor = '#5f5fff10';}

  return (
    <div {...bind()} style={{
      border: `0.3rem solid ${borderColor}`,
      backgroundColor: backgroundColor,
      position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', boxSizing: 'border-box', cursor: 'crosshair',
    }}></div>
  )
}


// 选出经过 camera 映射后, 会落入矩形 [xMin, xMax, yMin, yMax] 内的顶点
const getVerticesWithinRect = ({verticesPosRef: verticesPosRef, cameraRef, xMin, xMax, yMin, yMax}): Set<number> => {
  const vertices = new Set<number>();
  for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
    const pos3D = new Vector3(verticesPosRef.current[3 * idx], verticesPosRef.current[3 * idx + 1], verticesPosRef.current[3 * idx + 2]);
    const posScreen = pos3D.clone().project(cameraRef.current);   // 由于模型没有缩放旋转平移, 因此无需进行 mesh.matrixWorld 的变换
    if (posScreen.x >= xMin && posScreen.x <= xMax && posScreen.y >= yMin && posScreen.y <= yMax) {   // 落入矩形内
      vertices.add(idx);
    }
  }
  return vertices;
}


// 根据 faces 建立邻居表 (该表可以查询给定顶点 i 的全部邻居顶点 j)
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

// 根据 faces 建立边对顶点表 (该表可以查询给定边 (i, j) 的对顶点 k)
type OppositeVtxIdOfEdgeType = Map<string, number>;  // 形如 "123-456" -> 0
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

// 计算每个边 i-j 的权重 w_ij
type WIJType = Map<string, number>;  // 形如 "123-137" -> 0.5
const buildWij = (verticesPos: Float32Array, faces: Uint32Array, neighborTable: NeighborTableType, oppositeVtxIdOfEdge: OppositeVtxIdOfEdgeType): WIJType => {
  const wIJ = new Map();
  for (let vtxIdx = 0; vtxIdx < verticesPos.length / 3; vtxIdx++) {
    const neighborVtxIds = Array.from(neighborTable.get(vtxIdx));
    for (const neighborVtxId of neighborVtxIds) {

      // alpha_ij
      let cotAlpha;
      const edgeKeyIJ = `${vtxIdx}-${neighborVtxId}`;
      const oppositeVtxIdOfEdgeIJ = oppositeVtxIdOfEdge.get(edgeKeyIJ);
      if (oppositeVtxIdOfEdgeIJ !== undefined) { // 若该边是边界边, 则没有对顶点
        const vtxPosI = new Vector3(verticesPos[3 * vtxIdx], verticesPos[3 * vtxIdx + 1], verticesPos[3 * vtxIdx + 2]);
        const vtxPosJ = new Vector3(verticesPos[3 * neighborVtxId], verticesPos[3 * neighborVtxId + 1], verticesPos[3 * neighborVtxId + 2]);
        const vtxPosK = new Vector3(verticesPos[3 * oppositeVtxIdOfEdgeIJ], verticesPos[3 * oppositeVtxIdOfEdgeIJ + 1], verticesPos[3 * oppositeVtxIdOfEdgeIJ + 2]);
        const vecKI = new Vector3().subVectors(vtxPosK, vtxPosI).normalize();
        const vecKJ = new Vector3().subVectors(vtxPosK, vtxPosJ).normalize();
        const cosine = vecKI.dot(vecKJ);
        const sine = vecKI.clone().cross(vecKJ).length();
        cotAlpha = cosine / sine;
      }

      // beta_ij
      let cotBeta;
      const edgeKeyJI = `${neighborVtxId}-${vtxIdx}`;
      const oppositeVtxIdOfEdgeJI = oppositeVtxIdOfEdge.get(edgeKeyJI);
      if (oppositeVtxIdOfEdgeJI !== undefined) { // 若该边是边界边, 则没有对顶点
        const vtxPosI = new Vector3(verticesPos[3 * vtxIdx], verticesPos[3 * vtxIdx + 1], verticesPos[3 * vtxIdx + 2]);
        const vtxPosJ = new Vector3(verticesPos[3 * neighborVtxId], verticesPos[3 * neighborVtxId + 1], verticesPos[3 * neighborVtxId + 2]);
        const vtxPosK = new Vector3(verticesPos[3 * oppositeVtxIdOfEdgeJI], verticesPos[3 * oppositeVtxIdOfEdgeJI + 1], verticesPos[3 * oppositeVtxIdOfEdgeJI + 2]);
        const vecKI = new Vector3().subVectors(vtxPosK, vtxPosI).normalize();
        const vecKJ = new Vector3().subVectors(vtxPosK, vtxPosJ).normalize();
        const cosine = vecKI.dot(vecKJ);
        const sine = vecKI.clone().cross(vecKJ).length();
        cotBeta = cosine / sine;
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


// 根据 wIJ 构建系数矩阵 L
const buildMatrixL = (wIJ: WIJType, nrVertices: number, neighborTable: NeighborTableType): SparseMatrix => {
  const triplet = new Triplet(nrVertices, nrVertices);
  for (let i = 0; i < nrVertices; i++) {
    for (const j of neighborTable.get(i)) {
      const w_ij = wIJ.get(`${i}-${j}`);
      triplet.addEntry(w_ij, i, i);    // L_ii += w_ij
      triplet.addEntry(-w_ij, i, j);   // L_ij += (-w_ij)
    }
  }
  return SparseMatrix.fromTriplet(triplet);
}



const vertexTypeToColor = (type: VertexType): number => {
  switch (type) {
    case VertexType.Fixed:
      return 0xffaa55;
    case VertexType.Draggable:
      return 0xff0000;
    case VertexType.Calculated:
      return 0x049ef4;
  }
}


// 显示模型的组件
export const TriMesh = ({verticesPosRef: verticesPosRef, verticesOriginalRef, verticesTypeRef, cameraRef,
                          neighborTableRef, oppositeVtxIdOfEdgeRef, wIJRef, LMatrixRef}) => {

  const geometryRef = useRef<BufferGeometry>();
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);

  const [forceUpdate, setForceUpdate] = useRecoilState(ForceUpdateAtom);

  // 加载模型, 得到 vertices[] 和 faces[], 以及用于渲染的 geometry. (这个 useEffect 只会在初始化时执行一次)
  useEffect(() => {
    console.log('loading model')
    // 加载 obj 模型, 合并顶点, 转为 vertices 和 faces 数组
    const loader = new OBJLoader();
    loader.load('/person.obj', (object) => {
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

        // fixme: debug: 用于测试的固定顶点集、移动顶点集
        verticesTypeRef.current = new Array(vertices.length / 3).fill(VertexType.Calculated);
        for (const idx of [733, 734, 797, 798, 801, 802, 816, 817, 818, 820, 821, 822, 823, 824, 825, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 845, 847]) {
          verticesTypeRef.current[idx] = VertexType.Fixed;
        }
        for (const idx of [926, 927, 933, 934, 935, 936, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 980, 981, 982, 983, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062]) {
          verticesTypeRef.current[idx] = VertexType.Draggable;
        }

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

        // 允许 React 渲染
        setModelLoaded(true);

        // 回传给上一层组件的信息
        verticesPosRef.current = vertices;
        verticesOriginalRef.current = new Float32Array(vertices);
        geometryRef.current = geom;
        neighborTableRef.current = buildNeighborTable(faces);
        oppositeVtxIdOfEdgeRef.current = buildOppositeVtxIdOfEdge(faces);
        wIJRef.current = buildWij(vertices, faces, neighborTableRef.current, oppositeVtxIdOfEdgeRef.current);
        LMatrixRef.current = buildMatrixL(wIJRef.current, verticesPosRef.current.length / 3, neighborTableRef.current);

      });
    });
  }, []);

  // 设置相机位姿
  const {camera, gl, scene}: { camera: PerspectiveCamera } = useThree();
  const [cameraConfig, setCameraConfig] = useRecoilState(CamCfgAtom);
  cameraRef.current = camera;   // 回传给上一层, 计算框选用
  useEffect(() => {
    const {theta, phi, distance} = cameraConfig;
    const cameraPos = sphericalToCartesian({theta, phi, distance});
    camera.position.set(...cameraPos);
    camera.lookAt(0, 0, 0);
  })

  // 处理顶点拖拽
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
    const vecDelta = vecRight.clone().multiplyScalar(dRight).add(vecUp.clone().multiplyScalar(-dDown));
    // 更新拖拽指定的动点坐标
    for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
      if (verticesTypeRef.current[idx] === VertexType.Draggable) {
        verticesPosRef.current[3 * idx + 0] += vecDelta.x / sensitivity;
        verticesPosRef.current[3 * idx + 1] += vecDelta.y / sensitivity;
        verticesPosRef.current[3 * idx + 2] += vecDelta.z / sensitivity;
      }
    }
    // 计算非动点且非固定点的、符合 ARAP 的新坐标   fixme: 目前是设置为邻居节点的平均坐标 (just for fun/test)
    for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
      if (verticesTypeRef.current[idx] === VertexType.Calculated) {
        const table = neighborTableRef.current as NeighborTableType;
        const neighbors = Array(...table.get(idx)) as number[];
        const neighborsPos = neighbors.map(neiId => verticesPosRef.current.slice(3 * neiId, 3 * neiId + 3));
        const neighborsPosAvg = neighborsPos.reduce((acc, cur) => acc.map((v, i) => v + cur[i] / neighbors.length), [0, 0, 0]);
        verticesPosRef.current[3 * idx + 0] = neighborsPosAvg[0];
        verticesPosRef.current[3 * idx + 1] = neighborsPosAvg[1];
        verticesPosRef.current[3 * idx + 2] = neighborsPosAvg[2];
      }
    }
    // 通知 three 更新模型
    (geometryRef.current as BufferGeometry).attributes.position.needsUpdate = true;
    setForceUpdate(Math.random());  // 触发 points 重渲染, 但我不知道为啥这样就能解决..
  })


  return (
    <group {...bind()}>

      {/* 模型主体 */}
      {modelLoaded && (
        <mesh
          geometry={geometryRef.current}
          material={new MeshPhongMaterial({color: 0x049ef4})}
        />
      )}

      {/* 染色顶点 */}
      {modelLoaded && (
        verticesTypeRef.current.map((vtxType, vtxId) => (
          <mesh key={vtxId} position={new Array(...verticesPosRef.current.slice(3 * vtxId, 3 * vtxId + 3))}>
            <sphereGeometry args={[vtxType === VertexType.Calculated ? 0.002 : 0.004, 5, 5]}/>
            <meshPhongMaterial color={vertexTypeToColor(vtxType)}/>
          </mesh>
        ))
      )}

    </group>
  )
}




export const App = () => {

  // 相机参数
  const cameraRef = useRef();

  // 光源方向
  const [lightConfig, ] = useRecoilState(LightCfgAtom);
  const lightPos = sphericalToCartesian({theta: lightConfig.theta, phi: lightConfig.phi, distance: lightConfig.distance});

  // 顶点选择器类型: "none" | "drag" | "fixed" | "calc"
  const [selectorType, setSelectorType] = useState("none");

  // 所有顶点的 [变形前] 坐标
  const verticesOriginalRef = useRef<Float32Array>();

  // 所有顶点的 [当前(变形后)] 坐标
  const verticesPosRef = useRef<Float32Array>();

  // 每个顶点的状态: fixed | draggable | calculated
  const verticesTypeRef = useRef<VertexType[]>();

  // ARAP 算法中, 避免重复计算用
  const neighborTableRef = useRef<NeighborTableType>();         // 顶点邻接表
  const oppositeVtxIdOfEdgeRef = useRef<Map<string, number>>(); // 有向边 i->j 所在三角形的 ｢对点｣ 编号. 对于每个模型只需要构造一次
  const wIJRef = useRef<Map<string, number>>();             // w_ij: 边 i-j 的权重, 对于每个模型只需要计算一次
  const LMatrixRef = useRef<SparseMatrix>();                 // L: 系数矩阵, 对于每个模型只需要计算一次, 之后只需要取所需的 n 行 n 列即可
  const RiRef = useRef<Map<number, DenseMatrix>>();         // R_i: 顶点 i 的邻域 (Cell) 从原始位置 Ci 旋转到当前位置 Ci' 的旋转矩阵, 每当顶点 Ci' 移动时要更新


  return (
    <>
      <Canvas>

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
          verticesOriginalRef={verticesOriginalRef}
          verticesPosRef={verticesPosRef}
          verticesTypeRef={verticesTypeRef}
          cameraRef={cameraRef}
          neighborTableRef={neighborTableRef}
          oppositeVtxIdOfEdgeRef={oppositeVtxIdOfEdgeRef}
          wIJRef={wIJRef}
          LMatrixRef={LMatrixRef}
        />

      </Canvas>

      {/* 相机和光源参数控件 */}
      <CamPosCtl/>
      <LightPosCtl/>

      {/* 顶点交互框选器 */}
      <DraggableSelectorOn onClick={()=>{setSelectorType("drag")}}/>
      <FixedSelectorOn onClick={()=>{setSelectorType("fixed")}}/>
      <DeSelectorOn onClick={()=>{setSelectorType("calc")}}/>
      {selectorType !== "none" &&
        <RectSelector
          selectorType={selectorType}
          onDragDone={({xMin, xMax, yMin, yMax}) => {

            // 关闭顶点选择器
            setSelectorType("none");

            // 找出所有落入该矩形的顶点编号
            const verticesWithinRect = getVerticesWithinRect({verticesPosRef: verticesPosRef, cameraRef, xMin, xMax, yMin, yMax});

            // 将这些顶点设置为 Fixed 或 Draggable 或 Calc
            const types = verticesTypeRef.current as VertexType[];
            for (const idx of verticesWithinRect) {
              if (selectorType === "drag") {
                types[idx] = VertexType.Draggable;
              }
              else if (selectorType === "fixed") {
                types[idx] = VertexType.Fixed;
              }
              else if (selectorType === "calc") {
                types[idx] = VertexType.Calculated;
              }
            }

            // todo: 构建系数矩阵 L 和 b
            // 统计哪些顶点的坐标是待计算的
            const verticesToCalc = verticesTypeRef.current.map((vtxType, vtxId) => vtxType === VertexType.Calculated ? vtxId : -1).filter(vtxId => vtxId !== -1);
            const verticesSettled = verticesTypeRef.current.map((vtxType, vtxId) => vtxType !== VertexType.Calculated ? vtxId : -1).filter(vtxId => vtxId !== -1);
            console.log("verticesToCalc", verticesToCalc);
            console.log("verticesSettled", verticesSettled);

            // todo: 计算 R_i





          }}
        />
      }
    </>
  )
}


export default App;
