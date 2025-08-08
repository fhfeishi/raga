import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// src/pages/Chat.tsx
import React, { useState, useRef, useEffect } from 'react';
import AssistantBubble from '@/components/AssistantBubble';
import { useChatStore } from '@/stores/chatStore';
import styles from '@/pages/Chat.module.css';
const Chat = () => {
    const { addMessage, activeId, conversations, } = useChatStore();
    const activeConv = conversations.find(c => c.id === activeId);
    const messages = activeConv?.messages ?? [];
    const [input, setInput] = React.useState('');
    const [isStreaming, setIsStreaming] = useState(false); // 控制发送状态
    const textareaRef = useRef(null);
    const chatMainRef = useRef(null);
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
    const streamLLMResponse = async (question) => {
        setIsStreaming(true);
        // 创建一个临时的 assistant 消息，用于追加内容
        const assistantMsg = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: '',
            ts: Date.now(),
        };
        addMessage(activeId, assistantMsg);
        try {
            const response = await fetch(`/api/chat/stream?conversation_id=${activeId}&question=${encodeURIComponent(question)}`);
            if (!response.body)
                throw new Error('No response body');
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done)
                    break;
                buffer += decoder.decode(value, { stream: true });
                // 按换行解析 event
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // 剩余未完成的行
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]')
                            continue;
                        // 更新 assistant 消息内容
                        assistantMsg.content += data;
                        // 直接更新 store（需要 mutable 更新）
                        useChatStore.getState().updateMessageContent(activeId, assistantMsg.id, assistantMsg.content);
                    }
                }
            }
        }
        catch (err) {
            assistantMsg.content = `❌ 服务错误: ${err instanceof Error ? err.message : '未知错误'}`;
            useChatStore.getState().updateMessageContent(activeId, assistantMsg.id, assistantMsg.content);
        }
        finally {
            setIsStreaming(false);
        }
    };
    // 用户发送
    const handleSend = () => {
        if (!input.trim() || isStreaming)
            return;
        // 把用户消息加到当前会话
        addMessage(activeId, { role: 'user', content: input, ts: Date.now() });
        setInput('');
        // 流式获取 LLM 响应
        streamLLMResponse(input);
    };
    // 键盘事件
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };
    return (_jsxs("div", { className: styles.chatContainer, children: [_jsx("main", { className: styles.chatMain, ref: chatMainRef, children: messages.length === 0 ? (_jsx("div", { className: styles.emptyState, children: "\u5F00\u59CB\u4F60\u7684\u7B2C\u4E00\u6BB5\u5BF9\u8BDD\u5427\uFF01" })) : (messages.map(msg => (_jsx("div", { className: `${styles.messageBubble} ${msg.role === 'user' ? styles.userBubble : styles.assistantBubble}`, children: msg.role === 'assistant' ? (_jsx(AssistantBubble, { content: msg.content })) : (msg.content) }, msg.id)))) }), _jsxs("footer", { className: styles.chatFooter, children: [_jsx("textarea", { ref: textareaRef, value: input, onChange: e => setInput(e.target.value), onKeyDown: handleKeyDown, placeholder: "\u56DE\u8F66\u53D1\u9001\uFF0CShift+Enter \u6362\u884C", style: { resize: 'none', overflow: 'hidden' } }), _jsx("button", { onClick: handleSend, className: styles.sendBtn, children: "\u53D1\u9001" })] })] }));
};
export default Chat;
