// src/pages/Chat.tsx
import React, { useState, KeyboardEvent, useRef, useEffect } from 'react';
import AssistantBubble from '@/components/AssistantBubble';
import { useChatStore } from '@/stores/chatStore';
import styles from '@/pages/Chat.module.css';
import { Message } from '@/types'
import {v4 as uuidv4} from 'uuid';

const fastapi_router_chat = '/api/chat/stream'  // fastapi后端暴露的路由


const Chat: React.FC = () => {
  const {
    addMessage,
    activeId,
    conversations,
  } = useChatStore();

  const activeConv = conversations.find(c => c.id === activeId);
  const messages = activeConv?.messages ?? [];

  const [input, setInput] = React.useState('');
  const [isStreaming, setIsStreaming] = useState(false); // 控制发送状态
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const chatMainRef = useRef<HTMLDivElement | null>(null);

  // 自动调整 textarea 高度
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [input]);

  // 滚动到底部
  useEffect(() => {
    const chatmain = chatMainRef.current;
    if (chatmain) {
      chatmain.scrollTop = chatmain.scrollHeight;
    }
  }, [messages]);


  // 工具 把一个完整的 SSE 事件块解析出 data payload（可能多行 data:）
  const parseSSEEvent = (rawEvent: string): string | null => {
    // 一个事件块可能多行：id: xxx\n event: message\n data: ...\n data: ...\n\n
    // 这里把所有 data: 行拼起来（按 \n 连接）
    const lines = rawEvent.split(/\r?\n/);
    const datas: string[] = [];
    for (const ln of lines) {
      if (ln.startsWith('data:')) {
        datas.push(ln.slice(5).trimStart()); // 切掉 'data:' 前缀
      }
    }
    if (datas.length === 0) return null;
    // 有些服务端会发 JSON，这里先原样返回字符串，由上层决定怎么用
    return datas.join('\n');
  };

  // —— fallback：用 fetch 读流并自己按 “\n\n” 切事件
  const streamByFetch = async (url: string, onChunk: (text: string) => void) => {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'text/event-stream',
        // 如果需要 cookie： credentials: 'include'（注意 CORS）
      } as any,
    });
    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      // 以空行分割事件
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? '';

      for (const ev of events) {
        // 如果后端没加 data: 前缀（直接吐 token），这里兜底一下
        const hasDataPrefix = /^data:/m.test(ev);
        let payload: string | null;
        if (hasDataPrefix) {
          payload = parseSSEEvent(ev);
        } else {
          // 把整块当成 data
          payload = ev.trim();
        }
        if (!payload) continue;

        // 统一处理结束标志
        const p = payload.trim();
        if (p === '[END]' || p === '[DONE]') {
          onChunk('[__END__]');
        } else if (p.startsWith('[ERROR]')) {
          onChunk(`[__ERROR__] ${p.slice(7).trim()}`);
        } else {
          onChunk(payload);
        }
      }
    }
    // 流正常完结但未明确发 END，也补一个
    onChunk('[__END__]');
  };

  // —— 首选：EventSource（浏览器内置 SSE 客户端）
  const streamByEventSource = (url: string, onChunk: (text: string) => void) => {
    return new Promise<void>((resolve, reject) => {
      const es = new EventSource(url, {withCredentials: false});

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

      es.onerror = (e) => {
        // 很多时候 onerror 会先触发一次（CORS/网络），我们直接关闭并走 fallback
        es.close();
        reject(new Error('EventSource error'));
      };
    });
  };


  // 流式接收 LLM 响应
  const streamLLMResponse = async (question: string) => {

    if (!activeId) return;

    setIsStreaming(true);

    // 创建一个临时的 assistant 消息，用于追加内容
    const assistantMsg: Message = {
      // id: crypto.randomUUID(),
      id: uuidv4(),
      role: 'assistant',
      content: '',
      ts: Date.now(),
    };
    addMessage(activeId, assistantMsg);

    // 封装“写入内容”的函数
    const append = (delta: string) => {
      assistantMsg.content += delta;
      useChatStore.getState().updateMessageContent(activeId, assistantMsg.id, assistantMsg.content);
    };

    // 统一 URL（带上 conversation_id & question）
    const url = `${fastapi_router_chat}?conversation_id=${encodeURIComponent(activeId)}&question=${encodeURIComponent(question)}`;

    try {
      // 先尝试 EventSource（简单、可靠）
      await streamByEventSource(url, (text) => {
        if (text === '[__END__]') return; // 结束不再追加字符
        if (text.startsWith('[__ERROR__]')) {
          const msg = text.replace('[__ERROR__]', '❌ 服务错误:').trim();
          append(`\n${msg}\n`);
          return;
        }
        append(text);
      });
    } catch {
      // 若 EventSource 不可用或报错，则 fallback 到 fetch 解析
      try {
        await streamByFetch(url, (text) => {
          if (text === '[__END__]') return;
          if (text.startsWith('[__ERROR__]')) {
            const msg = text.replace('[__ERROR__]', '❌ 服务错误:').trim();
            append(`\n${msg}\n`);
            return;
          }
          append(text);
        });
      } catch (err: any) {
        const msg = err instanceof Error ? err.message : String(err);
        useChatStore.getState().updateMessageContent(
            activeId,
            assistantMsg.id,
            `❌ 服务错误: ${msg}`
        );
      }
    } finally {
      setIsStreaming(false);
    }
  };

  // 用户发送
  const handleSend = () => {
    if (!input.trim() || isStreaming || !activeId) return;
    // 把用户消息加到当前会话
    addMessage(activeId, {role: 'user', content: input, ts: Date.now()});

    const q = input;
    setInput('');

    // 流式获取 LLM 响应
    streamLLMResponse(q);
  };

  // 键盘事件
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
      <div className={styles.chatContainer}>
        {/* 对话列表 */}
        <main className={styles.chatMain} ref={chatMainRef}>
          {messages.length === 0 ? (
              <div className={styles.emptyState}>开始你的第一段对话吧！</div>
          ) : (
              messages.map(msg => (
                  <div
                      key={msg.id}
                      className={`${styles.messageBubble} ${
                          msg.role === 'user' ? styles.userBubble : styles.assistantBubble
                      }`}
                  >
                    {msg.role === 'assistant' ? (
                        <AssistantBubble content={msg.content}/>
                    ) : (
                        msg.content
                    )}
                  </div>
              ))
          )}
        </main>

        {/* 输入区域 */}
        <footer className={styles.chatFooter}>
        <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="回车发送，Shift+Enter 换行"
            style={{resize: 'none', overflow: 'hidden'}}
        />
          <button onClick={handleSend} className={styles.sendBtn} disabled={isStreaming || !activeId}>
            {isStreaming ? '生成中…' : '发送'}</button>
        </footer>
      </div>
  );
};

export default Chat;