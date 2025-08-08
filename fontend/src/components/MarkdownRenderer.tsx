// src/components/MarkdownRenderer.tsx
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css';

interface Props {
  source: string;
}

const MarkdownRenderer: React.FC<Props> = ({ source }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw, rehypeSanitize, rehypeHighlight]}
      components={{
        // 图片
        img: ({ ...props }) => (
          <img
            {...props}
            style={{
              maxWidth: '100%',
              height: 'auto',
              borderRadius: 8,
              marginTop: 8,
            }}
            alt={props.alt || 'image'}
          />
        ),
        // 引用
        blockquote: ({ children }) => (
          <blockquote
            style={{
              margin: '0.6em 0',
              paddingLeft: 12,
              borderLeft: '4px solid #ddd',
              color: '#666',
            }}
          >
            {children}
          </blockquote>
        ),
        // 链接
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: '#1677ff' }}>
            {children}
          </a>
        ),
        // 表格
        table: ({ children }) => (
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '0.6em', marginBottom: '0.6em' }}>
            {children}
          </table>
        ),
        th: ({ children }) => (
          <th style={{ border: '1px solid #ddd', padding: '6px', textAlign: 'left', backgroundColor: '#f5f5f5' }}>
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td style={{ border: '1px solid #ddd', padding: '6px' }}>
            {children}
          </td>
        ),
      }}
    >
      {source}
    </ReactMarkdown>
  );
};

export default MarkdownRenderer;