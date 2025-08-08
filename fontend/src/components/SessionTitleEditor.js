import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// src/components/SessionTitleEditor.tsx
import { useState, useRef, useEffect } from 'react';
import { useChatStore } from '@/stores/chatStore';
const SessionTitleEditor = ({ currentTitle, sessionId }) => {
    const [title, setTitle] = useState(currentTitle);
    const [showModal, setShowModal] = useState(false);
    const modalRef = useRef(null);
    const inputRef = useRef(null);
    const updateSessionTitle = useChatStore(state => state.updateSessionTitle);
    useEffect(() => {
        setTitle(currentTitle);
    }, [currentTitle]);
    const handleEditClick = () => {
        setShowModal(true);
    };
    const handleClose = () => {
        setShowModal(false);
        setTitle(currentTitle);
    };
    const handleConfirm = () => {
        if (sessionId && title.trim()) {
            updateSessionTitle(sessionId, title.trim());
        }
        setShowModal(false);
    };
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleConfirm();
        }
        else if (e.key === 'Escape') {
            handleClose();
        }
    };
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (modalRef.current && !modalRef.current.contains(e.target)) {
                handleClose();
            }
        };
        if (showModal) {
            document.addEventListener('mousedown', handleClickOutside);
            const timer = setTimeout(() => {
                inputRef.current?.focus();
            }, 100);
            return () => clearTimeout(timer);
        }
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showModal]);
    return (_jsxs("div", { style: { position: 'relative' }, children: [_jsx("button", { style: {
                    padding: '6px 12px',
                    backgroundColor: '#f0f0f0',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '16px',
                    fontSize: '14px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    minWidth: '100px',
                    textAlign: 'center',
                }, onClick: handleEditClick, title: "\u70B9\u51FB\u4FEE\u6539\u4F1A\u8BDD\u6807\u9898", children: title }), showModal && (_jsx("div", { style: {
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 1000,
                }, children: _jsxs("div", { ref: modalRef, style: {
                        background: '#fff',
                        borderRadius: '12px',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
                        width: '320px',
                        padding: '20px',
                        position: 'relative',
                    }, children: [_jsx("button", { onClick: handleClose, style: {
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
                            }, children: "\u00D7" }), _jsx("h3", { style: {
                                margin: '0 0 16px 0',
                                fontSize: '18px',
                                fontWeight: '500',
                                color: '#1a1a1a',
                                textAlign: 'center',
                            }, children: "\u4FEE\u6539\u5BF9\u8BDD\u6807\u7B7E" }), _jsx("input", { ref: inputRef, type: "text", value: title, onChange: e => setTitle(e.target.value), onKeyDown: handleKeyDown, style: {
                                width: '100%',
                                padding: '10px 12px',
                                border: '1px solid #ddd',
                                borderRadius: '8px',
                                fontSize: '14px',
                                outline: 'none',
                                boxSizing: 'border-box',
                            }, placeholder: "\u8BF7\u8F93\u5165\u65B0\u6807\u9898" }), _jsxs("div", { style: {
                                display: 'flex',
                                justifyContent: 'flex-end',
                                gap: '8px',
                                marginTop: '16px',
                            }, children: [_jsx("button", { onClick: handleConfirm, style: {
                                        width: '80px', // 固定宽度
                                        padding: '6px 0', // 水平留白由宽度控制，垂直留白保留
                                        background: '#1890ff',
                                        color: '#fff',
                                        border: 'none',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        cursor: 'pointer',
                                    }, children: "\u786E\u8BA4" }), _jsx("button", { onClick: handleClose, style: {
                                        width: '80px', // 固定宽度
                                        padding: '6px 0', // 水平留白由宽度控制，垂直留白保留
                                        background: '#fff',
                                        color: '#333',
                                        border: '1px solid #ddd',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        cursor: 'pointer',
                                    }, children: "\u53D6\u6D88" })] })] }) }))] }));
};
export default SessionTitleEditor;
