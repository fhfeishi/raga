import { jsx as _jsx } from "react/jsx-runtime";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css';
const MarkdownRenderer = ({ source }) => {
    return (_jsx(ReactMarkdown, { remarkPlugins: [remarkGfm], rehypePlugins: [rehypeRaw, rehypeSanitize, rehypeHighlight], components: {
            // 图片
            img: ({ ...props }) => (_jsx("img", { ...props, style: {
                    maxWidth: '100%',
                    height: 'auto',
                    borderRadius: 8,
                    marginTop: 8,
                }, alt: props.alt || 'image' })),
            // 引用
            blockquote: ({ children }) => (_jsx("blockquote", { style: {
                    margin: '0.6em 0',
                    paddingLeft: 12,
                    borderLeft: '4px solid #ddd',
                    color: '#666',
                }, children: children })),
            // 链接
            a: ({ href, children }) => (_jsx("a", { href: href, target: "_blank", rel: "noopener noreferrer", style: { color: '#1677ff' }, children: children })),
            // 表格
            table: ({ children }) => (_jsx("table", { style: { width: '100%', borderCollapse: 'collapse', marginTop: '0.6em', marginBottom: '0.6em' }, children: children })),
            th: ({ children }) => (_jsx("th", { style: { border: '1px solid #ddd', padding: '6px', textAlign: 'left', backgroundColor: '#f5f5f5' }, children: children })),
            td: ({ children }) => (_jsx("td", { style: { border: '1px solid #ddd', padding: '6px' }, children: children })),
        }, children: source }));
};
export default MarkdownRenderer;
