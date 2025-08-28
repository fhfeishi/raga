// src/pages/Chat.tsx
import React, { useState, KeyboardEvent, useRef, useEffect } from 'react';
import AssistantBubble from '@/components/AssistantBubble';
import { useChatStore } from '@/stores/chatStore';
import styles from '@/pages/Chat.module.css';
import { Message } from '@/types';
import { v4 as uuidv4 } from 'uuid';

//  简单开关：开发阶段设为 false，联调后端时改为 true
const BACKEND_ENABLED = false;

// 调试：可在控制台随时查看
if (typeof window !== 'undefined') {
  (window as any).BACKEND_ENABLED = BACKEND_ENABLED;
}

// 仅在需要后端时才用到
const fastapi_router_chat = '/api/chat/stream';

// 本地演示用答案
const demo_answer: string = `
以下 3 件最新公开的灵巧手相关专利，分别从“驱动-传动-感知”三大维度展示了目前行业最关注的创新点，可供后续研发快速借鉴。

* example-table:


| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |


## 1. 灵巧智能《一种三自由度腱绳驱动灵巧手指》公开号：CN1155xxxxxA（2025-03）
### 亮点速读
- 单指 3 DOF：第一、二指节由“微型电机＋涡轮蜗杆＋绞盘＋钢丝绳”间接驱动，第三指节由同轴电机直驱，实现“两级腱绳＋一级直驱”的混合传动，兼顾大抓握力与末端精细控制[引用1-原创力文档](https://max.book118.com/html/2024/0529/7064063015006114.shtm)
- 结构紧凑：所有执行器均藏在手掌内，手指本体外径≤14 mm，可直接替换现有夹爪末端。
- 低成本复用：钢丝绳采用标准 0.3 mm 航空级钢索，维修时无需拆整手，30 秒快拆更换。

![图1-图1的相关描述](/demoss/图1.jpg)
<p align="center">图1-图1的相关描述</p>


## 2. 新剑机电《无框力矩电机＋行星滚柱丝杠灵巧手》公开号：CN1178xxxxxA（2024-12）
### 亮点速读
- 15 DOF 全驱方案：每根手指均用一颗 20 mm 无框力矩电机驱动，通过行星滚柱丝杠将旋转运动变为直线，再经 3 级连杆放大为关节转角，整手仅需 5 颗电机即可实现 15 个主动自由度。[引用2-原创力·专利](https://zhuanli.book118.com/view/191212024fs25t2112421096.html)
- 高负载-低回差：丝杠导程 0.5 mm，理论传动效率 90%，在指尖可输出 5 kg 持续力而回差＜0.1°，满足工业插拔、拧紧等高精度场景。
- 模块化手指：拇指、食指可热插拔为 3 DOF 高灵活度模块，其余手指可替换 1 DOF 低成本模块，同一手掌兼容两种配置。

![图2-图2的相关描述](/demoss/图2.jpg)
<p align="center">图2-图2的相关描述</p>


## 3. 腾讯 Robotics X《TRX-Hand 刚柔混合驱动灵巧手》公开号：CN1169xxxxxA（2024-06）
### 亮点速读
- 刚柔混合驱动：8 个关节中 3 个采用“微型伺服电机＋谐波减速”刚性驱动，5 个采用“形状记忆合金弹簧＋柔性铰链”弹性驱动，既保证高速大负载（指尖 15 N、关节 600 °/s），又能在碰撞时通过柔性关节吸收能量，整机寿命提升 10 倍。[引用3-搜狐](https://www.sohu.com/a/670524247_320333)
-  全掌高密度感知：指尖、指腹、掌面共布置 240 点柔性触觉阵列＋1 颗微型激光雷达，实现 3 mm 分辨率、0.05 g 力变化检测，支持“盲抓”柔软物体。[引用4腾讯网](https://news.qq.com/rain/a/20230425A05J7700)
- 算法开源：配套发布 ROS2 驱动包和抓取数据集，开发者可直接调用 MoveIt! 和 YOLO-Grasp 模型完成二次开发。[引用5-知乎](https://zhuanlan.zhihu.com/p/625631528)

![图3-图3的相关描述](/demoss/图3.jpg)
<p align="center">图3-图3的相关描述</p>
`;

const Chat: React.FC = () => {
  const { addMessage, activeId, conversations } = useChatStore();
  const activeConv = conversations.find((c) => c.id === activeId);
  const messages = activeConv?.messages ?? [];

  // 仅在“接后端且正在流式”时用于禁用按钮/控制某条气泡的下载按钮
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingAssistantId, setStreamingAssistantId] = useState<string | null>(null);

  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const chatMainRef = useRef<HTMLDivElement | null>(null);

  // 自动调整 textarea 高度
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${ta.scrollHeight}px`;
  }, [input]);

  // 新消息滚动到底部
  useEffect(() => {
    const el = chatMainRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // ============= 流式接收（仅在 BACKEND_ENABLED === true 时生效） =============
  const streamLLMResponse = async (question: string) => {
    // 防御：在未接后端模式下被误调用，直接退出（不抛错、不发请求）
    if (!BACKEND_ENABLED) {
      console.warn('[Chat] streamLLMResponse called while BACKEND_ENABLED=false. Skipped.');
      return;
    }
    if (!activeId) return;

    setIsStreaming(true);

    // 插入一条空的 assistant 消息，后续 token 追加到这条
    const assistantMsg: Message = {
      id: uuidv4(),
      role: 'assistant',
      content: '',
      ts: Date.now(),
    };
    addMessage(activeId, assistantMsg);
    setStreamingAssistantId(assistantMsg.id);

    const setAssistantContent = (text: string) => {
      useChatStore.getState().updateMessageContent(activeId, assistantMsg.id, text);
    };
    const append = (delta: string) => {
      assistantMsg.content += delta;
      setAssistantContent(assistantMsg.content);
    };

    const url = `${fastapi_router_chat}?conversation_id=${encodeURIComponent(
        activeId
    )}&question=${encodeURIComponent(question)}`;

    let gotAnyToken = false;

    const parseSSEEvent = (rawEvent: string): string | null => {
      const lines = rawEvent.split(/\r?\n/);
      const datas: string[] = [];
      for (const ln of lines) {
        if (ln.startsWith('data:')) datas.push(ln.slice(5).trimStart());
      }
      return datas.length ? datas.join('\n') : null;
    };

    const onChunk = (text: string) => {
      if (text === '[__END__]') return;
      if (text.startsWith('[__ERROR__]')) {
        console.warn('[Chat] backend error chunk:', text);
        return;
      }
      gotAnyToken = true;
      append(text);
    };

    const streamByEventSource = () =>
        new Promise<void>((resolve, reject) => {
          const es = new EventSource(url, { withCredentials: false });
          es.onmessage = (evt) => {
            const payload = (evt.data ?? '').trim();
            if (!payload) return;
            if (payload === '[END]' || payload === '[DONE]') {
              onChunk('[__END__]');
              es.close();
              resolve();
            } else if (payload.startsWith('[ERROR]')) {
              onChunk(`[__ERROR__] ${payload.slice(7).trim()}`);
              es.close();
              resolve();
            } else {
              onChunk(payload);
            }
          };
          es.onerror = () => {
            es.close();
            reject(new Error('EventSource error'));
          };
        });

    const streamByFetch = async () => {
      const response = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'text/event-stream' } as any,
      });
      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() ?? '';

        for (const ev of events) {
          const hasDataPrefix = /^data:/m.test(ev);
          const payload = hasDataPrefix ? parseSSEEvent(ev) : ev.trim();
          if (!payload) continue;

          const p = payload.trim();
          if (p === '[END]' || p === '[DONE]') onChunk('[__END__]');
          else if (p.startsWith('[ERROR]')) onChunk(`[__ERROR__] ${p.slice(7).trim()}`);
          else onChunk(payload);
        }
      }
      onChunk('[__END__]');
    };

    try {
      await streamByEventSource();
    } catch (err) {
      console.warn('[Chat] EventSource failed, fallback to fetch:', err);
      try {
        await streamByFetch();
      } catch (err2) {
        console.warn('[Chat] fetch stream failed:', err2);
        // 后端失败兜底
        setAssistantContent(demo_answer.trim());
      }
    } finally {
      // 完全没收到 token → 兜底 demo
      if (!gotAnyToken || !assistantMsg.content.trim()) {
        console.log('[Chat] no tokens from backend, use demo_answer fallback.');
        setAssistantContent(demo_answer.trim());
      }
      setIsStreaming(false);
      setStreamingAssistantId(null);
    }
  };

  // ================================ 发送 ================================
  const handleSend = () => {
    if (!input.trim() || (BACKEND_ENABLED && isStreaming) || !activeId) return;

    // 1) 先把用户消息加入会话
    addMessage(activeId, { role: 'user', content: input, ts: Date.now() });

    // 2) 清空输入框
    const question = input;
    setInput('');

    // 3) 分支：不开后端 → 直接写 demo；接后端 → 走流式
    if (!BACKEND_ENABLED) {
      const assistantMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: demo_answer.trim(),
        ts: Date.now(),
      };
      addMessage(activeId, assistantMsg);
      return;
    }

    streamLLMResponse(question);
  };

  // Enter 发送，Shift+Enter 换行
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ================================ 渲染 ================================
  return (
      <div className={styles.chatContainer}>
        <main className={styles.chatMain} ref={chatMainRef}>
          {messages.length === 0 ? (
              <div className={styles.emptyState}>开始你的第一段对话吧！</div>
          ) : (
              messages.map((msg) => (
                  <div
                      key={msg.id}
                      className={`${styles.messageBubble} ${
                          msg.role === 'user' ? styles.userBubble : styles.assistantBubble
                      }`}
                  >
                    {msg.role === 'assistant' ? (
                        <AssistantBubble
                            content={msg.content}
                            // 只有“正在流的这条”视为未完成需要隐藏下载；不开后端时永远视为完成
                            isComplete={!BACKEND_ENABLED || streamingAssistantId !== msg.id}
                            fileBaseName="answer"
                        />
                    ) : (
                        msg.content
                    )}
                  </div>
              ))
          )}
        </main>

        <footer className={styles.chatFooter}>
        <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="回车发送，Shift+Enter 换行"
            style={{ resize: 'none', overflow: 'hidden' }}
        />
          <button
              onClick={handleSend}
              className={styles.sendBtn}
              // 只有“接后端且正在流式”时禁用；不开后端保持可点击
              disabled={(BACKEND_ENABLED && isStreaming) || !activeId}
              title={!activeId ? '请先选择或新建一个会话' : undefined}
          >
            {BACKEND_ENABLED && isStreaming ? '生成中…' : '发送'}
          </button>
        </footer>
      </div>
  );
};

export default Chat;
