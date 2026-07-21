import {
  AssistantRuntimeProvider,
  type AppendMessage,
  type ThreadMessageLike,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { type ReactNode, useCallback } from "react";

import type { ChatMessage } from "../lib/types";

type AgentRuntimeProviderProps = {
  children: ReactNode;
  messages: ChatMessage[];
  isRunning: boolean;
  onSend: (content: string) => Promise<void>;
  onCancel: () => Promise<void>;
};

const convertMessage = (message: ChatMessage): ThreadMessageLike => ({
  id: message.id,
  role: message.role,
  content: [{ type: "text", text: message.content }],
  createdAt: new Date(message.created_at),
});

export function AgentRuntimeProvider({
  children,
  messages,
  isRunning,
  onSend,
  onCancel,
}: AgentRuntimeProviderProps) {
  const handleNew = useCallback(
    async (message: AppendMessage) => {
      const content = message.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();
      if (content) await onSend(content);
    },
    [onSend],
  );

  // 原因：会话和消息以 Qwopus SQLite 为唯一数据源，assistant-ui 不应再保存一份影子历史。
  // 作用：ExternalStoreRuntime 只负责专业聊天交互，所有发送与取消仍进入 FastAPI。
  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage,
    isRunning,
    onNew: handleNew,
    onCancel,
  });

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>;
}
