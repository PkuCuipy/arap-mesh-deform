import { useEffect, useRef, useState } from 'react';
import { useDrag } from 'react-use-gesture';

import * as THREE from "three";
import { Box3, BufferAttribute, BufferGeometry, Mesh, MeshPhongMaterial, Object3D, PerspectiveCamera, Vector3 } from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Box, Sky } from "@react-three/drei";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";

import { useRecoilState } from 'recoil';
import { CamCfgAtom, LightCfgAtom, SelectorCfgAtom } from './atoms';
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
export const MovingSelectorOn = () => {
  const [selectorConfig, setSelectorConfig] = useRecoilState(SelectorCfgAtom);
  const handleClick = () => {
    setSelectorConfig((prev) => ({selectorType: "moving", hidden: false}));
  }
  return (
    <div className={'moving-selector-on-button'} onClick={handleClick}>Moving Select</div>
  )
};


// 定点选择器开启组件
export const FixedSelectorOn = () => {
  const [selectorConfig, setSelectorConfig] = useRecoilState(SelectorCfgAtom);
  const handleClick = () => {
    setSelectorConfig((prev) => ({selectorType: "fixed", hidden: false}));
  }
  return (
    <div className={'fixed-selector-on-button'} onClick={handleClick}>Fixed Select</div>
  )
};


// 显示模型的组件
export const TriMesh = ({verticesRef}) => {

  const materialRef = useRef<MeshPhongMaterial>();
  const verticesOriginalRef = useRef<Float32Array>();
  const facesRef = useRef<Uint32Array>();
  const geometryRef = useRef<BufferGeometry>();
  const meshRef = useRef<Object3D>();
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const {camera, gl, scene}: { camera: PerspectiveCamera } = useThree();

  // 设置相机位姿
  const [cameraConfig, setCameraConfig] = useRecoilState(CamCfgAtom);
  useEffect(() => {
    const {theta, phi, distance} = cameraConfig;
    const cameraPos = sphericalToCartesian({theta, phi, distance});
    camera.position.set(...cameraPos);
    camera.lookAt(0, 0, 0);
  })

  // 被固定的顶点集
  const fixedIndicesRef: number[] = useRef();
  const fixedPositionsRef: Vector3[] = useRef();

  // 交互移动的顶点集
  const movingIndicesRef: number[] = useRef();
  const movingPositionsRef: Vector3[] = useRef();

  // Doc: https://use-gesture.netlify.app/docs/state/
  const sensitivity = 100;
  const bind = useDrag(({delta: [dRight, dDown]}) => {
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

      for (const vtxIdx of movingIndicesRef.current) {
        movingPositionsRef.current[vtxIdx].add(new Vector3(
          vecDelta.x / sensitivity,
          vecDelta.y / sensitivity,
          vecDelta.z / sensitivity
        ))
        verticesRef.current[3 * vtxIdx + 0] = movingPositionsRef.current[vtxIdx].x
        verticesRef.current[3 * vtxIdx + 1] = movingPositionsRef.current[vtxIdx].y
        verticesRef.current[3 * vtxIdx + 2] = movingPositionsRef.current[vtxIdx].z
      }
      geometryRef.current.attributes.position.needsUpdate = true; // 通知 three 更新模型

    }
  )


  // 加载模型, 得到 vertices[] 和 faces[], 以及用于渲染的 geometry. (这个 useEffect 只会在初始化时执行一次)
  useEffect(() => {
    // 设置材质
    materialRef.current = new MeshPhongMaterial({
      color: 0x049ef4,
      flatShading: false,
    });
    // 加载 obj 模型, 合并顶点, 转为 vertices 和 faces 数组
    const loader = new OBJLoader();
    loader.load('/person.obj', (object) => {
      object.traverse((child) => {
        if (!(child instanceof Mesh))
          return;
        // 用 mergeVertices() 合并相同的顶点 (不知道为啥读取的模型是 non-indexed 的)
        child.geometry.deleteAttribute('normal');   // 如果不删除, 就没法 merge
        child.geometry.deleteAttribute('uv');
        child.geometry = mergeVertices(child.geometry);
        // 构建 geometry 数据结构
        const vertices = child.geometry.attributes.position.array as Float32Array;
        const faces = child.geometry.index.array as Uint32Array;
        const geom = new BufferGeometry();
        geom.setAttribute("position", new BufferAttribute(vertices, 3));
        geom.setIndex(new BufferAttribute(faces, 1));
        geom.computeVertexNormals();
        // 将模型居中, 使得 bbox 的中心点为 [0,0,0]. (注: 通过修改 verticesRef.current 实现)
        geom.center();
        // 将模型的 bbox 缩放到 [-0.5, 0.5] 立方体内. (注: 通过修改 verticesRef.current 实现)
        const bboxSize = (new Box3()).setFromBufferAttribute(geom.attributes.position).getSize(new Vector3());
        const maxSize = Math.max(bboxSize.x, bboxSize.y, bboxSize.z);
        const scale = 1 / maxSize;
        geom.scale(scale, scale, scale);

        // 触发 React 渲染
        setModelLoaded(true);

        // 将 geom, vertices 和 faces 存为 Ref 变量. (vertices 和 faces 其实是 geom 中的字段)
        verticesRef.current = vertices;
        verticesOriginalRef.current = new Float32Array(vertices);
        facesRef.current = faces;
        geometryRef.current = geom;

        // debug: 用于测试的固定顶点集、移动顶点集
        movingIndicesRef.current = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        movingPositionsRef.current = movingIndicesRef.current.map((idx) => new Vector3(
          verticesRef.current[3 * idx + 0],
          verticesRef.current[3 * idx + 1],
          verticesRef.current[3 * idx + 2])
        );
        fixedIndicesRef.current = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        fixedPositionsRef.current = fixedIndicesRef.current.map((idx) => new Vector3(
          verticesRef.current[3 * idx + 0],
          verticesRef.current[3 * idx + 1],
          verticesRef.current[3 * idx + 2])
        );

      });
    });
  }, []);



  return (
    <>
      {modelLoaded && (
        <mesh
          {...bind()}
          ref={meshRef}
          geometry={geometryRef.current}
          material={materialRef.current}
        />)}
    </>
  );
}






// 矩形顶点选择器
export const RectSelector = ({hidden, selectorType, }) => {

  const [selectorConfig, setSelectorConfig] = useRecoilState(SelectorCfgAtom);

  const bind = useDrag(({initial: [x0, y0], movement: [x, y], first, last}) => {

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
    }
    if (last) {
      console.log("end dragging")
      console.log(xMin, xMax, "-->", yMin, yMax);
      setSelectorConfig({hidden: true});
    }

  })
  return (
    <div {...bind()} style={{
      display: hidden ? 'none' : 'block',
      border: selectorType === "moving" ? '0.3rem solid #ff5f9f' : '0.3rem solid #ffdf5f',
      backgroundColor: selectorType === "moving" ? '#ffdfdf30' : '#ffdf5f10',
      position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', boxSizing: 'border-box', cursor: 'crosshair',
    }}></div>
  )
}


export const App = () => {

  const [lightConfig, ] = useRecoilState(LightCfgAtom);
  const lightPos = sphericalToCartesian({theta: lightConfig.theta, phi: lightConfig.phi, distance: lightConfig.distance});

  const [selectorConfig, ] = useRecoilState(SelectorCfgAtom);

  const verticesRef = useRef<Float32Array>();

  return (
    <>
      <Canvas>

        {/* 天空和光源 */}
        <Sky/>
        <ambientLight intensity={0.3}/>
        <pointLight position={lightPos} intensity={0.7}/>

        {/* 模型 */}
        <TriMesh
          verticesRef={verticesRef}
        />

      </Canvas>

      {/* 相机和光源参数控件 */}
      <CamPosCtl/>
      <LightPosCtl/>

      {/* 顶点交互框选器 */}
      <MovingSelectorOn/>
      <FixedSelectorOn/>
      <RectSelector
        hidden={selectorConfig.hidden}
        selectorType={selectorConfig.selectorType}
      />

    </>
  )
}


export default App;
