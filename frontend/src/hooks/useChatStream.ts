import { useState, useCallback, useRef } from 'react';

export type Role = 'user' | 'assistant';

export interface ChatMessage {
    id: string;
    role: Role;
    content: string;
    isStreaming?: boolean;
}

export function useChatStream() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // We use an abort controller so we can cancel requests if needed.
    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = useCallback(async (content: string) => {
        if (!content.trim() || isGenerating) return;

        // Reset error
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
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: content.trim() }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch: ${response.statusText}`);
            }

            const reader = response.body?.getReader();
            const decoder = new TextDecoder('utf-8');

            if (!reader) {
                throw new Error('No readable stream available.');
            }

            let done = false;
            let fullResponse = '';
            let buffer = '';

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;

                if (value) {
                    buffer += decoder.decode(value, { stream: true });

                    let lineEnd = buffer.indexOf('\n\n');
                    while (lineEnd !== -1) {
                        const eventStr = buffer.slice(0, lineEnd).trim();
                        buffer = buffer.slice(lineEnd + 2);

                        if (eventStr.startsWith('data: ')) {
                            const data = eventStr.slice(6);
                            if (data === '[DONE]') {
                                done = true;
                                break;
                            }
                            if (data.startsWith('[ERROR]')) {
                                throw new Error(data.slice(7));
                            }

                            // Parse JSON safely
                            try {
                                const parsed = JSON.parse(data);
                                fullResponse += parsed;
                            } catch (e) {
                                fullResponse += data;
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

            // Mark streaming as false
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
                const errorMessage = err instanceof Error ? err.message : 'An error occurred during Generation.';
                setError(errorMessage);
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === aiMessageId
                            ? { ...msg, content: msg.content + '\n\n*(Error connecting to server)*', isStreaming: false }
                            : msg
                    )
                );
            }
        } finally {
            setIsGenerating(false);
            abortControllerRef.current = null;
        }
    }, [isGenerating]);

    const stopGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
    }, []);

    return {
        messages,
        isGenerating,
        error,
        sendMessage,
        stopGeneration,
    };
}
