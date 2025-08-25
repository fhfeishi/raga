// src/types.d.ts
/// <reference types="vite/client" />
// 引入 vite 的环境变量和 HMR 类型
// 在编译这个项目的时候，把 vite/client 这个模块的类型声明文件也引进来。


declare module '*.module.css' {
    const classes: { readonly [key: string]: string };
    export default classes;
  }

declare module '*.svg?react' {
    import React from 'react';
    const content: React.FC<React.SVGProps<SVGSVGElement>>;
    export default content;
}