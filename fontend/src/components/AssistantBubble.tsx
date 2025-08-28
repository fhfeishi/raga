// src/components/AssistantBubble.tsx
import React, { useCallback, useEffect, useRef, useState, useId } from 'react';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import { saveAs } from 'file-saver';
import * as docx from 'docx';
import Download from '@/assets/icons/downloada.svg?react';
import styles from '@/components/AssistantBubble.module.css';

interface Props {
  /** 本气泡内要展示/导出的 Markdown 文本 */
  content: string;
  /** 该气泡是否已“流式完成”。未完成则不显示下载入口 */
  isComplete?: boolean;
  /** 是否显示下载入口（即使 complete）。缺省 true */
  showDownload?: boolean;
  /** 导出文件名前缀（不含后缀），缺省 'answer' */
  fileBaseName?: string;
  /** 点击下载时回调（可选：埋点/统计） */
  onDownload?: (format: 'md' | 'docx') => void;
}

const AssistantBubble: React.FC<Props> = ({
                                            content,
                                            isComplete = true,
                                            showDownload = true,
                                            fileBaseName = 'answer',
                                            onDownload,
                                          }) => {
  // —— 每个气泡**独立**的菜单开关状态（互不影响）
  const [menuOpen, setMenuOpen] = useState(false);

  // —— 这些 ref 只属于当前气泡
  const menuRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  // —— 生成无冲突的 id，方便 a11y 绑定 aria-controls
  const menuId = useId();

  // —— 是否显示下载按钮（所有条件就绪才显示）
  const canShowDownload =
      showDownload && isComplete && content.trim().length > 0;

  // —— 文件名规范化（避免非法字符）
  const normalizedBaseName = fileBaseName
      .trim()
      .replace(/[\\/:*?"<>|]/g, '_') || 'answer';

  // 保存 Markdown（纯文本直存）
  const saveMd = useCallback(() => {
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    saveAs(blob, `${normalizedBaseName}.md`);
    setMenuOpen(false);
    onDownload?.('md');
  }, [content, normalizedBaseName, onDownload]);

  // 保存 Word（将每一行转换为一个段落，避免整段黏在一起）
  const saveDocx = useCallback(async () => {
    const lines = content.replace(/\r\n/g, '\n').split('\n');

    const children = lines.map(
        (line) =>
            new docx.Paragraph({
              children: [new docx.TextRun(line || ' ')], // 空行也保留
            })
    );

    const document = new docx.Document({
      sections: [{ properties: {}, children }],
    });

    const blob = await docx.Packer.toBlob(document);
    saveAs(blob, `${normalizedBaseName}.docx`);
    setMenuOpen(false);
    onDownload?.('docx');
  }, [content, normalizedBaseName, onDownload]);

  // —— 打开后：点击气泡外部/按 Esc 关闭菜单（只监听当前气泡）
  useEffect(() => {
    if (!menuOpen) return;

    const onDocClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (menuRef.current?.contains(t) || btnRef.current?.contains(t)) return;
      setMenuOpen(false);
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };

    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  // —— 如果父组件把该气泡切回“未完成”（极少见），则收起菜单以避免状态错乱
  useEffect(() => {
    if (!isComplete && menuOpen) setMenuOpen(false);
  }, [isComplete, menuOpen]);

  return (
      <div className={styles.bubble}>
        {/* 渲染 Markdown 内容 */}
        <MarkdownRenderer source={content} />

        {/* 右下角下载图标与菜单（仅当本气泡已完成且内容非空） */}
        {canShowDownload && (
            <div className={styles.downloadContainer}>
              <button
                  ref={btnRef}
                  id={`${menuId}-trigger`}
                  type="button"
                  className={styles.downloadButton}
                  aria-haspopup="menu"
                  aria-expanded={menuOpen}
                  aria-controls={`${menuId}-menu`}
                  aria-label="下载此答案"
                  title="下载此答案"
                  onClick={() => setMenuOpen((v) => !v)}
              >
                <Download className={styles.downloadIcon} />
              </button>

              {menuOpen && (
                  <div
                      ref={menuRef}
                      id={`${menuId}-menu`}
                      role="menu"
                      aria-labelledby={`${menuId}-trigger`}
                      aria-label="下载为"
                      className={styles.downloadMenu}
                      // 阻止菜单内部点击触发外层关闭
                      onMouseDown={(e) => e.stopPropagation()}
                  >
                    <button
                        role="menuitem"
                        className={styles.menuItem}
                        onClick={saveMd}
                    >
                      保存为 .md
                    </button>
                    <button
                        role="menuitem"
                        className={styles.menuItem}
                        onClick={saveDocx}
                    >
                      保存为 .docx
                    </button>
                  </div>
              )}
            </div>
        )}
      </div>
  );
};

export default AssistantBubble;
