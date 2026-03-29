import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Bot, User } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type Role = 'user' | 'assistant';

interface MessageBubbleProps {
  role: Role;
  content: string;
  isStreaming?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ role, content, isStreaming }) => {
  const isUser = role === 'user';

  return (
    <div className={cn("flex w-full mt-4 space-x-3 max-w-3xl mx-auto", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <Avatar className="w-8 h-8 flex-shrink-0 mt-1 ring-1 ring-indigo-500/30">
          <AvatarFallback className="bg-gradient-to-br from-indigo-500/20 to-indigo-900/40 text-indigo-200">
            <Bot size={18} />
          </AvatarFallback>
        </Avatar>
      )}

      <Card
        className={cn(
          "px-5 py-3.5 text-[0.95rem] shadow-sm leading-relaxed overflow-hidden",
          isUser
            ? "bg-indigo-600/90 text-white rounded-2xl rounded-tr-sm border-indigo-500/50"
            : "bg-background/40 backdrop-blur-md rounded-2xl rounded-tl-sm border-white/5",
          "relative"
        )}
      >
        <div className={cn("prose prose-sm max-w-none dark:prose-invert", isUser && "text-white prose-p:text-white pb-0")}>
          {isUser ? (
            <p className="whitespace-pre-wrap m-0">{content}</p>
          ) : (
            <>
              {content ? (
                <ReactMarkdown
                  components={{
                    p: ({ node: _node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                    ul: ({ node: _node, ...props }) => <ul className="my-2 ml-4 list-disc marker:text-indigo-400" {...props} />,
                    ol: ({ node: _node, ...props }) => <ol className="my-2 ml-4 list-decimal marker:text-indigo-400" {...props} />,
                    li: ({ node: _node, ...props }) => <li className="pl-1 mb-1" {...props} />,
                    strong: ({ node: _node, ...props }) => <strong className="font-semibold text-indigo-100" {...props} />,
                  }}
                >
                  {content}
                </ReactMarkdown>
              ) : (
                <div className="flex space-x-1.5 items-center h-5 px-1">
                  <span className="w-1.5 h-1.5 bg-indigo-400/80 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-indigo-400/80 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-indigo-400/80 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              )}
              {isStreaming && content && (
                 <span className="ml-[1px] inline-block w-2 h-4 bg-indigo-500 animate-pulse align-middle opacity-80" />
              )}
            </>
          )}
        </div>
      </Card>

      {isUser && (
        <Avatar className="w-8 h-8 flex-shrink-0 mt-1 border border-indigo-700">
          <AvatarFallback className="bg-indigo-900 text-indigo-100">
            <User size={18} />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
};
