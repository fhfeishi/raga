// src/App.tsx
import React, { useState, useEffect } from 'react';
import SideBar from '@/components/SideBar';
import Chat from '@/pages/Chat';      // 聊天内容组件
import { useChatStore } from '@/stores/chatStore';  // 会话状态管理（Zustand）
import { useUiStore } from '@/stores/uiStore';   // UI 状态管理（如侧边栏展开/收起）
import NewChat from '@/assets/icons/new_chat.svg?react';
import SideButton from '@/assets/icons/side.svg?react';
import SessionTitleEditor from '@/components/SessionTitleEditor'; // 修改会话标签
import '@/styles.css';

const App: React.FC = () => {
  // 从 UI Store 获取侧边栏状态和切换函数
  const { isSidebarOpen, toggleSidebar } = useUiStore();
  // 从 Chat Store 获取会话数据和操作方法
  const { conversations, activeId, addConversation, updateSessionTitle } = useChatStore();

  // ---------------- 当前会话标题 ----------------
  // 找到当前激活的会话
  const currentSession = conversations.find((c) => c.id === activeId);
  const currentTitle = currentSession?.title || '新会话';
  // 使用本地状态管理输入框值（受控组件）
  const [title, setTitle] = useState(currentTitle);

  // 当 currentTitle 变化时（如切换会话），同步更新本地状态
  useEffect(() => {
    setTitle(currentTitle);
  }, [currentTitle]);

  // 处理标题输入框变化
  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTitle = e.target.value;
    setTitle(newTitle); // 更新输入框显示
    // 如果当前会话存在，同步更新到全局状态
    if (currentSession) updateSessionTitle(currentSession.id, newTitle);
  };

  return (
      <div className="app">
        {/* ========== 收起状态：全局固定标题栏 ========== */}
        {!isSidebarOpen && (
            <header className="app-header collapsed">
              <button onClick={toggleSidebar} title="展开导航">
                <SideButton className="w-5 h-5" />
              </button>
              <button onClick={addConversation} title="新建会话">
                <NewChat className="w-5 h-5" />
              </button>
              <SessionTitleEditor currentTitle={currentTitle} sessionId={currentSession?.id} />
            </header>
        )}

        {/* ========== 主体区域 ========== */}
        <div className="app-body">
          {isSidebarOpen && <SideBar />}
          <main className="chat-center">
            {/* ========== 展开状态：聊天区内标题栏 ========== */}
            {isSidebarOpen && (
                <div className="chat-header expanded">
                  <SessionTitleEditor currentTitle={currentTitle} sessionId={currentSession?.id} />
                </div>
            )}

            {/*/!* ---------- 主体区域 ---------- *!/*/}
            {/*<div className="app-body">*/}
            {/*  /!* 左侧边栏 *!/*/}
            {/*  {isSidebarOpen && <SideBar />}*/}

            {/*  /!* 右侧聊天区 *!/*/}
            {/*  <main className="chat-center" >*/}
            {/*    /!* 情况1：侧边栏收起 → 标题栏固定在顶部 *!/*/}
            {/*    {!isSidebarOpen &&(*/}
            {/*        <header className="app-header collapsed">*/}
            {/*          <button onClick={toggleSidebar} title="展开导航">*/}
            {/*            <SideButton className="w-5 h-5" />*/}
            {/*          </button>*/}
            {/*          <button onClick={addConversation} title={"新建会话"}>*/}
            {/*            <NewChat className="w-5 h-5"/>*/}
            {/*          </button>*/}
            {/*          <SessionTitleEditor currentTitle={currentTitle} sessionId={currentSession?.id} />*/}
            {/*        </header>*/}
            {/*    )}*/}
            {/*    /!* 情况2：侧边栏展开 → 标题栏在聊天区内顶部 *!/*/}
            {/*    {isSidebarOpen && (*/}
            {/*        <div className="chat-header expanded">*/}
            {/*          <SessionTitleEditor currentTitle={currentTitle} sessionId={currentSession?.id} />*/}
            {/*        </div>*/}
            {/*    )}*/}
            <Chat />
          </main>
        </div>
      </div>
  );
};

export default App;