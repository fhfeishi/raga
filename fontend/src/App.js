import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// src/App.tsx
import { useState, useEffect } from 'react';
import SideBar from '@/components/SideBar';
import Chat from '@/pages/Chat'; // 聊天内容组件
import { useChatStore } from '@/stores/chatStore'; // 会话状态管理（Zustand）
import { useUiStore } from '@/stores/uiStore'; // UI 状态管理（如侧边栏展开/收起）
import NewChat from '@/assets/icons/new_chat.svg?react';
import SideButton from '@/assets/icons/side.svg?react';
import SessionTitleEditor from '@/components/SessionTitleEditor'; // 修改会话标签
import '@/styles.css';
const App = () => {
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
    const handleTitleChange = (e) => {
        const newTitle = e.target.value;
        setTitle(newTitle); // 更新输入框显示
        // 如果当前会话存在，同步更新到全局状态
        if (currentSession)
            updateSessionTitle(currentSession.id, newTitle);
    };
    return (_jsxs("div", { className: "app", children: [!isSidebarOpen && (_jsxs("header", { className: "app-header collapsed", children: [_jsx("button", { onClick: toggleSidebar, title: "\u5C55\u5F00\u5BFC\u822A", children: _jsx(SideButton, { className: "w-5 h-5" }) }), _jsx("button", { onClick: addConversation, title: "\u65B0\u5EFA\u4F1A\u8BDD", children: _jsx(NewChat, { className: "w-5 h-5" }) }), _jsx(SessionTitleEditor, { currentTitle: currentTitle, sessionId: currentSession?.id })] })), _jsxs("div", { className: "app-body", children: [isSidebarOpen && _jsx(SideBar, {}), _jsxs("main", { className: "chat-center", children: [isSidebarOpen && (_jsx("div", { className: "chat-header expanded", children: _jsx(SessionTitleEditor, { currentTitle: currentTitle, sessionId: currentSession?.id }) })), _jsx(Chat, {})] })] })] }));
};
export default App;
