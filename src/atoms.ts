import { atom } from 'recoil';

export const CamCfgAtom = atom({
  key: 'CamCfgAtom',
  default: {
    phi: Math.PI / 2,
    theta: 0,
    distance: 1,
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

export const SelectorCfgAtom = atom({
  key: 'SelectorCfgAtom',
  default: {
    hidden: true,
    selectorType: null,
  }
})
