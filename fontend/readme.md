

```python
~ fontend/
    ├── public/             # resuorces
    ├── src/                # code
    │   ├──assets/          # resources 
    │   ├──components/       
    │   │    ├── AssistantBubble.tsx            # 智能体会话气泡
    │   │    ├── AssistantBubble.module.css
    │   │    ├── MarkdownRenderer.tsx           # 气泡内markdown文本渲染逻辑
    │   │    ├── SessionTitleEditor.tsx         # 会话窗口的标题更新
    │   │    ├── SideBar.module.css            
    │   │    └── SideBar.tsx                    # 侧边栏（展开、收起）按钮
    │   ├── hooks/
    │   │    └── useSSE.ts                      # 流式文本 no use
    │   ├── pages/
    │   │    ├── Chat.module.css
    │   │    └── Chat.tsx                       # 用户-智能体会话逻辑
    │   ├── stores/
    │   │    ├── chatStore.ts                   # 
    │   │    └── usStore.ts
    │   ├── App.tsx
    │   ├── index.tsx
    │   ├── styles.css                          
    │   ├── types.ts                             
    │   └── vite-env.d.ts                       # '*.svg?react'、'*.module.css'
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── tsconfig.json
    ├── vite.config.ts
    └── readme.md
```


```bash

# 快速删掉误生成的 .js/.js.map, src/下
# windows
(path/to/fontend)$ Get-ChildItem -Path src -Include *.js,*.js.map -Recurse | Remove-Item

# linux/macos
(path/to/fontend)$ find src -type f \( -name "*.js" -o -name "*.js.map" \) -delete

```



