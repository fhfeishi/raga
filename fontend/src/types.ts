// src/types.ts
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  ts: number; // 添加时间戳字段
}
export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  ts: number; // 创建时间戳
}