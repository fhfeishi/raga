// src/stores/uiStore.ts    ui状态管理
import { create } from 'zustand';
export const useUiStore = create((set) => ({
    isSidebarOpen: true,
    toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
}));
