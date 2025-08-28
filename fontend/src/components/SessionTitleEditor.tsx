// src/components/SessionTitleEditor.tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useChatStore } from '@/stores/chatStore';

interface Props {
    currentTitle: string;
    sessionId?: string;
}

const MIN_LEN = 1;
const MAX_LEN = 80;

const SessionTitleEditor: React.FC<Props> = ({ currentTitle, sessionId }) => {
    const updateSessionTitle = useChatStore(s => s.updateSessionTitle);

    const [title, setTitle] = useState(currentTitle);
    const [showModal, setShowModal] = useState(false);
    const modalRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // 仅在未打开弹窗时，同步外部标题，避免覆盖正在编辑的内容
    useEffect(() => {
        if (!showModal) setTitle(currentTitle);
    }, [currentTitle, showModal]);

    const open = useCallback(() => {
        setShowModal(true);
    }, []);
    const close = useCallback(() => {
        setShowModal(false);
        setTitle(currentTitle); // 还原未保存的更改
    }, [currentTitle]);

    const valid = !!sessionId && title.trim().length >= MIN_LEN && title.trim().length <= MAX_LEN;

    const confirm = useCallback(() => {
        if (!valid) return;
        updateSessionTitle(sessionId!, title.trim());
        setShowModal(false);
    }, [sessionId, title, valid, updateSessionTitle]);

    const onKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                confirm();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                close();
            }
        },
        [confirm, close]
    );

    // 打开弹窗时：注册外点点击、聚焦并全选、锁定滚动；统一清理
    useEffect(() => {
        if (!showModal) return;

        const handleClickOutside = (e: MouseEvent) => {
            if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
                close();
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        const prevOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';

        const t = window.setTimeout(() => {
            if (inputRef.current) {
                inputRef.current.focus();
                inputRef.current.select();
            }
        }, 0);

        return () => {
            window.clearTimeout(t);
            document.removeEventListener('mousedown', handleClickOutside);
            document.body.style.overflow = prevOverflow;
        };
    }, [showModal, close]);

    return (
        <div style={{ position: 'relative' }}>
            <button
                type="button"
                onClick={open}
                title={sessionId ? '点击修改会话标题' : '暂无可编辑会话'}
                aria-label="编辑会话标题"
                disabled={!sessionId}
                style={{
                    padding: '6px 12px',
                    backgroundColor: '#f0f0f0',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '16px',
                    fontSize: '14px',
                    cursor: sessionId ? 'pointer' : 'not-allowed',
                    transition: 'all 0.2s ease',
                    minWidth: '100px',
                    textAlign: 'center',
                    opacity: sessionId ? 1 : 0.6,
                }}
            >
                {title}
            </button>

            {showModal && (
                <div
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="dialog-label"
                    style={{
                        position: 'fixed',
                        inset: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.5)',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        zIndex: 1000,
                    }}
                    // 遮罩点击关闭（避免事件穿透）
                    onMouseDown={(e) => {
                        // 若点击直接发生在遮罩（而不是内容）上，也关闭
                        if (e.target === e.currentTarget) close();
                    }}
                >
                    <div
                        ref={modalRef}
                        style={{
                            background: '#fff',
                            borderRadius: '12px',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
                            width: '340px',
                            padding: '20px',
                            position: 'relative',
                        }}
                        onMouseDown={(e) => e.stopPropagation()} // 阻止冒泡到遮罩
                    >
                        <button
                            type="button"
                            onClick={close}
                            aria-label="关闭"
                            style={{
                                position: 'absolute',
                                top: '12px',
                                right: '12px',
                                width: '28px',
                                height: '28px',
                                border: 'none',
                                background: '#f0f0f0',
                                borderRadius: '50%',
                                fontSize: '18px',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: '#999',
                            }}
                        >
                            ×
                        </button>

                        <h3
                            id="dialog-label"
                            style={{
                                margin: '0 0 16px 0',
                                fontSize: '18px',
                                fontWeight: 500,
                                color: '#1a1a1a',
                                textAlign: 'center',
                            }}
                        >
                            修改对话标签
                        </h3>

                        <form
                            onSubmit={(e) => {
                                e.preventDefault();
                                confirm();
                            }}
                        >
                            <input
                                ref={inputRef}
                                type="text"
                                value={title}
                                onChange={(e) => setTitle(e.target.value)}
                                onKeyDown={onKeyDown}
                                maxLength={MAX_LEN}
                                placeholder="请输入新标题"
                                aria-invalid={!valid}
                                style={{
                                    width: '100%',
                                    padding: '10px 12px',
                                    border: '1px solid #ddd',
                                    borderRadius: '8px',
                                    fontSize: '14px',
                                    outline: 'none',
                                    boxSizing: 'border-box',
                                }}
                            />

                            <div
                                style={{
                                    display: 'flex',
                                    justifyContent: 'flex-end',
                                    gap: '8px',
                                    marginTop: '16px',
                                    alignItems: 'center',
                                }}
                            >
                <span style={{ fontSize: 12, color: '#999', marginRight: 'auto' }}>
                  {title.trim().length}/{MAX_LEN}
                </span>

                                <button
                                    type="submit"
                                    disabled={!valid}
                                    style={{
                                        width: '80px',
                                        padding: '6px 0',
                                        background: valid ? '#1890ff' : '#9ec9ff',
                                        color: '#fff',
                                        border: 'none',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        cursor: valid ? 'pointer' : 'not-allowed',
                                    }}
                                >
                                    确认
                                </button>
                                <button
                                    type="button"
                                    onClick={close}
                                    style={{
                                        width: '80px',
                                        padding: '6px 0',
                                        background: '#fff',
                                        color: '#333',
                                        border: '1px solid #ddd',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        cursor: 'pointer',
                                    }}
                                >
                                    取消
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SessionTitleEditor;
