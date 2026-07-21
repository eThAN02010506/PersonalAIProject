import {
  ActionBarPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { ArrowUp, Bot, Copy, Square } from "lucide-react";
import remarkGfm from "remark-gfm";

const MarkdownText = () => (
  <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="message-markdown" />
);

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message-row assistant-message">
      <div className="assistant-avatar" aria-hidden="true">
        <Bot size={17} />
      </div>
      <div className="message-body">
        <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
        <ActionBarPrimitive.Root className="message-actions" hideWhenRunning>
          <ActionBarPrimitive.Copy className="icon-button subtle" title="Copy response">
            <Copy size={15} />
          </ActionBarPrimitive.Copy>
        </ActionBarPrimitive.Root>
      </div>
    </MessagePrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="message-row user-message">
      <div className="user-bubble">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

export function ChatThread() {
  return (
    <ThreadPrimitive.Root className="thread-root">
      <ThreadPrimitive.Viewport className="thread-viewport">
        <ThreadPrimitive.Empty>
          <div className="empty-thread">
            <Bot size={25} />
            <h1>Qwopus Agent</h1>
            <p>What are we working on?</p>
          </div>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
        <div className="composer-dock">
          <ComposerPrimitive.Root className="composer-root">
            <ComposerPrimitive.Input
              className="composer-input"
              placeholder="Message Qwopus Agent"
              rows={1}
              autoFocus
            />
            <ComposerPrimitive.Send className="composer-action" title="Send message">
              <ArrowUp size={18} />
            </ComposerPrimitive.Send>
            <ComposerPrimitive.Cancel className="composer-action cancel" title="Stop response">
              <Square size={14} fill="currentColor" />
            </ComposerPrimitive.Cancel>
          </ComposerPrimitive.Root>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
