import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import MarkdownRenderer from '@/components/MarkdownRenderer';
import { saveAs } from 'file-saver'; // 轻量下载库
import * as docx from 'docx';
import styles from '@/components/AssistantBubble.module.css';
const AssistantBubble = ({ content }) => {
    // 保存 Markdown
    const saveMd = () => {
        const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
        saveAs(blob, 'answer.md');
    };
    // 保存 Word
    const saveDocx = async () => {
        const doc = new docx.Document({
            sections: [
                {
                    properties: {},
                    children: [
                        new docx.Paragraph({
                            text: content,
                            style: 'BodyText',
                        }),
                    ],
                },
            ],
        });
        const buffer = await docx.Packer.toBlob(doc);
        saveAs(buffer, 'answer.docx');
    };
    return (_jsxs("div", { className: styles.bubble, children: [_jsx(MarkdownRenderer, { source: content }), _jsxs("div", { className: styles.actions, children: [_jsx("button", { onClick: saveMd, style: { marginRight: 6 }, children: "\u4FDD\u5B58\u4E3A .md" }), _jsx("button", { onClick: saveDocx, children: "\u4FDD\u5B58\u4E3A .docx" })] })] }));
};
export default AssistantBubble;
