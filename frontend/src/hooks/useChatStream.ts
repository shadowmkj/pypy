import { useState, useCallback, useRef } from 'react';

export type Role = 'user' | 'assistant';

export interface ChatMessage {
    id: string;
    role: Role;
    isStreaming?: boolean;
    content: string;
}

interface UseChatStreamOptions {
    initialSessionId?: number | null;
    onSessionCreated?: (sessionId: number) => void;
}

export function useChatStream(options: UseChatStreamOptions = {}) {
    const { initialSessionId = null, onSessionCreated } = options;

    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<number | null>(initialSessionId);

    const abortControllerRef = useRef<AbortController | null>(null);

    /**
     * Load a past session's messages into state (called when user picks a session from sidebar).
     */
    const loadMessages = useCallback(
        (
            rawMessages: { id: number; isResponse: boolean; content: string }[],
            newSessionId: number
        ) => {
            setSessionId(newSessionId);
            setError(null);
            setMessages(
                rawMessages.map((m) => ({
                    id: m.id.toString(),
                    role: m.isResponse ? 'assistant' : 'user',
                    content: m.content,
                }))
            );
        },
        []
    );

    /** Reset to a fresh chat (new session). */
    const resetChat = useCallback(() => {
        setMessages([]);
        setSessionId(null);
        setError(null);
        abortControllerRef.current?.abort();
    }, []);

    const sendMessage = useCallback(
        async (content: string) => {
            if (!content.trim() || isGenerating) return;

            setError(null);
            setIsGenerating(true);

            const userMessage: ChatMessage = {
                id: Date.now().toString(),
                role: 'user',
                content: content.trim(),
            };

            const aiMessageId = (Date.now() + 1).toString();
            const initialAiMessage: ChatMessage = {
                id: aiMessageId,
                role: 'assistant',
                content: '',
                isStreaming: true,
            };

            setMessages((prev) => [...prev, userMessage, initialAiMessage]);

            abortControllerRef.current = new AbortController();

            try {
                const response = await fetch('/api/chat/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: content.trim(), sessionId }),
                    signal: abortControllerRef.current.signal,
                });

                if (!response.ok) {
                    throw new Error(`Request failed: ${response.statusText}`);
                }

                const reader = response.body?.getReader();
                const decoder = new TextDecoder('utf-8');

                if (!reader) throw new Error('No readable stream available.');

                let done = false;
                let fullResponse = '';
                let buffer = '';
                let currentEvent = ''; // tracks custom event type

                while (!done) {
                    const { value, done: readerDone } = await reader.read();
                    done = readerDone;

                    if (value) {
                        buffer += decoder.decode(value, { stream: true });

                        let lineEnd = buffer.indexOf('\n\n');
                        while (lineEnd !== -1) {
                            const eventStr = buffer.slice(0, lineEnd).trim();
                            buffer = buffer.slice(lineEnd + 2);

                            // Parse multi-line SSE block
                            const lines = eventStr.split('\n');
                            let eventType = currentEvent;
                            let dataLine = '';

                            for (const line of lines) {
                                if (line.startsWith('event: ')) {
                                    eventType = line.slice(7).trim();
                                } else if (line.startsWith('data: ')) {
                                    dataLine = line.slice(6);
                                }
                            }
                            currentEvent = ''; // reset after each block

                            // ── Handle `event: session` ──────────────────────
                            if (eventType === 'session') {
                                try {
                                    const parsed = JSON.parse(dataLine);
                                    if (parsed.sessionId) {
                                        setSessionId(parsed.sessionId);
                                        if (parsed.isNew && onSessionCreated) {
                                            onSessionCreated(parsed.sessionId);
                                        }
                                    }
                                } catch { /* ignore */ }
                                lineEnd = buffer.indexOf('\n\n');
                                continue;
                            }

                            // ── Handle regular data frames ───────────────────
                            if (dataLine === '[DONE]') {
                                done = true;
                                break;
                            }
                            if (dataLine.startsWith('[ERROR]')) {
                                throw new Error(dataLine.slice(7));
                            }

                            if (dataLine) {
                                try {
                                    const parsed = JSON.parse(dataLine);
                                    fullResponse += parsed;
                                } catch {
                                    fullResponse += dataLine;
                                }

                                setMessages((prev) =>
                                    prev.map((msg) =>
                                        msg.id === aiMessageId
                                            ? { ...msg, content: fullResponse }
                                            : msg
                                    )
                                );
                            }

                            lineEnd = buffer.indexOf('\n\n');
                        }
                    }
                }

                // Mark streaming complete
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === aiMessageId
                            ? { ...msg, content: fullResponse, isStreaming: false }
                            : msg
                    )
                );
            } catch (err: unknown) {
                if (err instanceof Error && err.name === 'AbortError') {
                    console.log('Stream aborted.');
                } else {
                    const errorMessage =
                        err instanceof Error ? err.message : 'An error occurred during generation.';
                    setError(errorMessage);
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === aiMessageId
                                ? {
                                      ...msg,
                                      content: msg.content + '\n\n*(Error connecting to server)*',
                                      isStreaming: false,
                                  }
                                : msg
                        )
                    );
                }
            } finally {
                setIsGenerating(false);
                abortControllerRef.current = null;
            }
        },
        [isGenerating, sessionId, onSessionCreated]
    );

    const stopGeneration = useCallback(() => {
        abortControllerRef.current?.abort();
    }, []);

    return {
        messages,
        isGenerating,
        error,
        sessionId,
        setSessionId,
        sendMessage,
        loadMessages,
        resetChat,
        stopGeneration,
    };
}
