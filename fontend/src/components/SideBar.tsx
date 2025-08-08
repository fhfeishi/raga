// src/components/SideBar.tsx
import React from 'react';
import { useChatStore } from '@/stores/chatStore';       // 会话数据
import { useUiStore } from '@/stores/uiStore';           // UI状态
import dayjs from 'dayjs';                               // 事件处理库
import SideButton from '@/assets/icons/side.svg?react';  // 侧边栏 展开/收起 按钮图标
import NewChat from '@/assets/icons/new_chat.svg?react'; // 新建会话 图标
import Logo from '@/assets/icons/logoa.svg?react';      // logo
import styles from '@/components/SideBar.module.css';               

// SideBar 组件
const SideBar: React.FC = () => {
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
    week: conversations.filter(
      (c) => dayjs(c.ts).isAfter(now.subtract(7, 'day')) && !dayjs(c.ts).isSame(now, 'day')
    ),
    // 更早
    earlier: conversations.filter((c) => dayjs(c.ts).isBefore(now.subtract(7, 'day'))),
  } as const;

  // ---------------- 折叠态 ----------------
  // 如果侧边栏关闭，返回一个占位元素（宽度与展开时一致，避免布局跳动）
  if (!isSidebarOpen) {
    return (
      <aside className={styles.sidebar} data-collapsed="true">
        <div className={styles.collapsedPlaceholder} />
      </aside>
    );
  }

  // ---------------- 展开态 ----------------
  return (
    <aside className={styles.sidebar} data-collapsed="false">
      {/* 顶部：Logo + 应用标题 + 收起按钮 */}
      <div className={styles.header}>
        <div className={styles.logoRow}>
          <Logo className="w-6 h-6" />
          <h1 className={styles.appTitle}>RAG智能体</h1>
        </div>
        {/* 点击收起侧边栏 */}
        <button className={styles.collapseBtn} onClick={toggleSidebar} title="收起导航">
          <SideButton className="w-5 h-5" />
        </button>
      </div>

      {/* 新建会话按钮（整行） */}
      <div className={styles.actionRow}>
        <button className={styles.newConversationBtn} onClick={addConversation} title="新建会话">
          <NewChat className="w-5 h-5" />
          <span>新建会话</span>
        </button>
      </div>

      {/* 会话列表 */}
      <nav className={styles.nav}>
        {/* 遍历分组 */}
        {Object.entries(groups).map(([key, list]) =>
          // 如果该组有会话才渲染
          list.length ? (
            <section key={key} className={styles.section}>
              {/* 分组标题 */}
              <h3>{key === 'today' ? '今天' : key === 'week' ? '7天内' : '更早'}</h3>
              <ul>
                {/* 会话项列表 */}
                {list.map((c) => (
                  <li
                    key={c.id}
                    // 动态类名：如果是当前会话，添加 active 样式
                    className={`${styles.item} ${c.id === activeId ? styles.active : ''}`}
                    // 点击切换会话
                    onClick={() => setActive(c.id)}
                    // 悬停显示完整标题
                    title={c.title}
                  >
                    {c.title}
                  </li>
                ))}
              </ul>
            </section>
          ) : null
        )}
      </nav>
    </aside>
  );
};

export default SideBar;