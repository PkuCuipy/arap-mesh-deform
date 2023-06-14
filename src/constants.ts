import { atom } from 'recoil';

export const CamCfgAtom = atom({
  key: 'CamCfgAtom',
  default: {
    phi: Math.PI / 2,
    theta: 0,
    distance: 2,
  }
})

export const LightCfgAtom = atom({
  key: 'LightCfgAtom',
  default: {
    phi: Math.PI / 2,
    theta: 0,
    distance: 10,
  }
})

// 顶点类型: fixed | draggable | calculated (default)
export const enum VertexType {
  Calculated = 0,
  Fixed = 1,
  Draggable = 2,
}
export const vertexTypeToColor = (type: VertexType): number => {
  switch (type) {
    case VertexType.Fixed:
      return 0xffaa55;
    case VertexType.Draggable:
      return 0xff0000;
    case VertexType.Calculated:
      return 0x049ef4;
  }
}


export const ForceUpdateAtom = atom({
  key: 'ForceUpdateAtom',
  default: 0.0,
})