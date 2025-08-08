// src/types.d.ts
/// <reference types="vite/client" />

declare module '*.module.css' {
    const classes: { readonly [key: string]: string };
    export default classes;
  }

declare module '*.svg?react' {
    import React from 'react';
    const content: React.FC<React.SVGProps<SVGSVGElement>>;
    export default content;
}