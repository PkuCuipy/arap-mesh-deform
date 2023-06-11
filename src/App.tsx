import { useEffect, useRef, useState } from 'react';
import { useDrag } from 'react-use-gesture';

import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Mesh,
  MeshPhongMaterial,
  PerspectiveCamera,
  PointsMaterial,
  Vector3
} from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Points, Sky } from "@react-three/drei";
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
  return (
    <div {...bind()} style={{
      border: selectorType === "drag" ? '0.3rem solid #ff5f9f' : '0.3rem solid #ffdf5f',
      backgroundColor: selectorType === "drag" ? '#ffdfdf30' : '#ffdf5f10',
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
type NeighborTable = Map<number, Set<number>>;
const buildNeighborTable = (faces: Uint32Array): NeighborTable => {
  const neighborTable = new Map();
  for (let faceIdx = 0; faceIdx < faces.length / 3; faceIdx++) {
    const [va, vb, vc] = faces.slice(3 * faceIdx, 3 * faceIdx + 3);
    if (!neighborTable.has(va)) {
      neighborTable.set(va, new Set());
    }
    if (!neighborTable.has(vb)) {
      neighborTable.set(vb, new Set());
    }
    if (!neighborTable.has(vc)) {
      neighborTable.set(vc, new Set());
    }
    neighborTable.get(va).add(vb);
    neighborTable.get(va).add(vc);
    neighborTable.get(vb).add(va);
    neighborTable.get(vb).add(vc);
    neighborTable.get(vc).add(va);
    neighborTable.get(vc).add(vb);
  }
  return neighborTable;
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
export const TriMesh = ({verticesPosRef: verticesPosRef, verticesOriginalRef, verticesTypeRef, cameraRef, neighborTableRef}) => {

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
        console.log('neighborTable', neighborTableRef.current);
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
    // 1. 更新拖拽指定的动点坐标
    for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
      if (verticesTypeRef.current[idx] === VertexType.Draggable) {
        verticesPosRef.current[3 * idx + 0] += vecDelta.x / sensitivity;
        verticesPosRef.current[3 * idx + 1] += vecDelta.y / sensitivity;
        verticesPosRef.current[3 * idx + 2] += vecDelta.z / sensitivity;
      }
    }
    // 2. 计算非动点且非固定点的、符合 ARAP 的新坐标
    // fixme: 设置为邻居节点的平均坐标 (just for fun/test)

    for (let idx = 0; idx < verticesPosRef.current.length / 3; idx++) {
      if (verticesTypeRef.current[idx] === VertexType.Calculated) {
        const table = neighborTableRef.current as NeighborTable;
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

  // 顶点选择器类型: "none" | "drag" | "fixed"
  const [selectorType, setSelectorType] = useState("none");

  // 所有顶点的 [变形前] 坐标
  const verticesOriginalRef = useRef<Float32Array>();

  // 所有顶点的 [当前(变形后)] 坐标
  const verticesPosRef = useRef<Float32Array>();

  // 每个顶点的状态: fixed | draggable | calculated
  const verticesTypeRef = useRef<VertexType[]>();

  // 顶点邻接表
  const neighborTableRef = useRef<NeighborTable>();


  return (
    <>
      <Canvas>

        {/* 天空和光源 */}
        <Sky/>
        <ambientLight intensity={0.1}/>
        <pointLight position={lightPos} intensity={0.7}/>
        <pointLight position={[-13, 100, 3]} intensity={0.2}/>
        <pointLight position={[4, -100, 11]} intensity={0.2}/>
        <pointLight position={[100, 1, -7]} intensity={0.2}/>
        <pointLight position={[-100, -3, 9]} intensity={0.2}/>
        <pointLight position={[11, 4, -100]} intensity={0.2}/>
        <pointLight position={[-12, 14, 100]} intensity={0.2}/>

        {/* 模型 */}
        <TriMesh
          verticesOriginalRef={verticesOriginalRef}
          verticesPosRef={verticesPosRef}
          verticesTypeRef={verticesTypeRef}
          cameraRef={cameraRef}
          neighborTableRef={neighborTableRef}
        />

      </Canvas>

      {/* 相机和光源参数控件 */}
      <CamPosCtl/>
      <LightPosCtl/>

      {/* 顶点交互框选器 */}
      <DraggableSelectorOn onClick={()=>{setSelectorType("drag")}}/>
      <FixedSelectorOn onClick={()=>{setSelectorType("fixed")}}/>
      {selectorType !== "none" &&
        <RectSelector
          selectorType={selectorType}
          onDragDone={({xMin, xMax, yMin, yMax}) => {
            setSelectorType("none");
            // console.log("选取了矩形:", xMin, xMax, yMin, yMax);
            const verticesWithinRect = getVerticesWithinRect({verticesPosRef: verticesPosRef, cameraRef, xMin, xMax, yMin, yMax});
            // console.log("框选的顶点编号:", verticesWithinRect);
            if (selectorType === "drag") {
              verticesWithinRect.forEach(idx => {
                (verticesTypeRef.current as VertexType[])[idx] = VertexType.Draggable;
              })
            } else if (selectorType === "fixed") {
              verticesWithinRect.forEach(idx => {
                (verticesTypeRef.current as VertexType[])[idx] = VertexType.Fixed;
              })
            }
          }}
        />
      }
    </>
  )
}


export default App;
