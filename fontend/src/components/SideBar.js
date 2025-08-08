import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useChatStore } from '@/stores/chatStore'; // 会话数据
import { useUiStore } from '@/stores/uiStore'; // UI状态
import dayjs from 'dayjs'; // 事件处理库
import SideButton from '@/assets/icons/side.svg?react'; // 侧边栏 展开/收起 按钮图标
import NewChat from '@/assets/icons/new_chat.svg?react'; // 新建会话 图标
import Logo from '@/assets/icons/logoa.svg?react'; // logo
import styles from '@/components/SideBar.module.css';
// SideBar 组件
const SideBar = () => {
    // 获取 UI 状态
    const { isSidebarOpen, toggleSidebar } = useUiStore();
    // 获取会话数据和操作
    const { conversations, activeId, setActive, addConversation } = useChatStore();
    // ---------------- 时间分组 ----------------
    const now = dayjs();
    // 将会话按时间分组
    const groups = {
        // 今天
        today: conversations.filter((c) => dayjs(c.ts).isSame(now, 'day')),
        // 过去 7 天内（不含今天）
        week: conversations.filter((c) => dayjs(c.ts).isAfter(now.subtract(7, 'day')) && !dayjs(c.ts).isSame(now, 'day')),
        // 更早
        earlier: conversations.filter((c) => dayjs(c.ts).isBefore(now.subtract(7, 'day'))),
    };
    // ---------------- 折叠态 ----------------
    // 如果侧边栏关闭，返回一个占位元素（宽度与展开时一致，避免布局跳动）
    if (!isSidebarOpen) {
        return (_jsx("aside", { className: styles.sidebar, "data-collapsed": "true", children: _jsx("div", { className: styles.collapsedPlaceholder }) }));
    }
    // ---------------- 展开态 ----------------
    return (_jsxs("aside", { className: styles.sidebar, "data-collapsed": "false", children: [_jsxs("div", { className: styles.header, children: [_jsxs("div", { className: styles.logoRow, children: [_jsx(Logo, { className: "w-6 h-6" }), _jsx("h1", { className: styles.appTitle, children: "RAG\u667A\u80FD\u4F53" })] }), _jsx("button", { className: styles.collapseBtn, onClick: toggleSidebar, title: "\u6536\u8D77\u5BFC\u822A", children: _jsx(SideButton, { className: "w-5 h-5" }) })] }), _jsx("div", { className: styles.actionRow, children: _jsxs("button", { className: styles.newConversationBtn, onClick: addConversation, title: "\u65B0\u5EFA\u4F1A\u8BDD", children: [_jsx(NewChat, { className: "w-5 h-5" }), _jsx("span", { children: "\u65B0\u5EFA\u4F1A\u8BDD" })] }) }), _jsx("nav", { className: styles.nav, children: Object.entries(groups).map(([key, list]) => 
                // 如果该组有会话才渲染
                list.length ? (_jsxs("section", { className: styles.section, children: [_jsx("h3", { children: key === 'today' ? '今天' : key === 'week' ? '7天内' : '更早' }), _jsx("ul", { children: list.map((c) => (_jsx("li", { 
                                // 动态类名：如果是当前会话，添加 active 样式
                                className: `${styles.item} ${c.id === activeId ? styles.active : ''}`, 
                                // 点击切换会话
                                onClick: () => setActive(c.id), 
                                // 悬停显示完整标题
                                title: c.title, children: c.title }, c.id))) })] }, key)) : null) })] }));
};
export default SideBar;
