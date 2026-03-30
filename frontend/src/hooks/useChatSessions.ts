import { useState, useEffect, useCallback } from 'react';

export interface ChatSessionMeta {
    id: number;
    title: string;
    updatedAt: string;
    lastMessage: {
        content: string;
        isResponse: boolean;
        createdAt: string;
    } | null;
}

export function useChatSessions() {
    const [sessions, setSessions] = useState<ChatSessionMeta[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchSessions = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch('/api/chat/sessions');
            if (!res.ok) throw new Error('Failed to load sessions');
            const data: ChatSessionMeta[] = await res.json();
            setSessions(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchSessions();
    }, [fetchSessions]);

    return { sessions, isLoading, error, refresh: fetchSessions };
}
