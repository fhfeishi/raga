// src/pages/Chat.tsx
import React, { useState, KeyboardEvent, useRef, useEffect } from 'react';
import AssistantBubble from '@/components/AssistantBubble';
import { useChatStore } from '@/stores/chatStore';
import styles from '@/pages/Chat.module.css';
import { Message } from '@/types'


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
  const textareaRef = useRef<HTMLTextAreaElement | null > (null);
  const chatMainRef = useRef<HTMLDivElement | null > (null);

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

  // 流式接收 LLM 响应
  const streamLLMResponse = async (question: string) => {
    setIsStreaming(true);

    // 创建一个临时的 assistant 消息，用于追加内容
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      ts: Date.now(),
    };
    addMessage(activeId!, assistantMsg);

    try {
      const response = await fetch(`/api/chat/stream?conversation_id=${activeId}&question=${encodeURIComponent(question)}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // 按换行解析 event
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 剩余未完成的行

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            // 更新 assistant 消息内容
            assistantMsg.content += data;
            // 直接更新 store（需要 mutable 更新）
            useChatStore.getState().updateMessageContent(activeId!, assistantMsg.id, assistantMsg.content);
          }
        }
      }
    } catch (err) {
      assistantMsg.content = `❌ 服务错误: ${err instanceof Error ? err.message : '未知错误'}`;
      useChatStore.getState().updateMessageContent(activeId!, assistantMsg.id, assistantMsg.content);
    } finally {
      setIsStreaming(false);
    }
  };

  // 用户发送
  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    // 把用户消息加到当前会话
    addMessage(activeId!, { role: 'user',  content: input , ts: Date.now() });
    setInput('');

    // 流式获取 LLM 响应
    streamLLMResponse(input);
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
        ): (
          messages.map(msg => (
            <div
              key={msg.id}
              className={`${styles.messageBubble} ${
                msg.role === 'user' ? styles.userBubble : styles.assistantBubble
              }`}
            >
              {msg.role === 'assistant' ? (
                <AssistantBubble content={msg.content} />
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
          style={{ resize: 'none', overflow: 'hidden' }}
        />
        <button onClick={handleSend} className={styles.sendBtn}>发送</button>
      </footer>
    </div>
  );
};

export default Chat;