'use client';

import React, { useState, useRef, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Send, Square, Mic, MicOff } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { useVoiceInput } from '@/hooks/useVoiceInput';

interface ChatInputProps {
    onSendMessage: (message: string) => void;
    onStop: () => void;
    isGenerating: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
    onSendMessage,
    onStop,
    isGenerating,
}) => {
    const [input, setInput] = useState('');
    // Holds the committed text before voice input started, so we can show interim results
    const committedInputRef = useRef('');

    const { isListening, isSupported, startListening, stopListening } = useVoiceInput({
        onInterimTranscript: (interim) => {
            // Show interim (live) text appended to the committed text
            setInput(committedInputRef.current + interim);
        },
        onFinalTranscript: (final) => {
            // Append the final recognized text (with a space separator)
            const prefix = committedInputRef.current;
            const separator = prefix && !prefix.endsWith(' ') ? ' ' : '';
            committedInputRef.current = prefix + separator + final;
            setInput(committedInputRef.current);
        },
    });

    const handleMicToggle = () => {
        if (isListening) {
            stopListening();
        } else {
            // Snapshot current input before speaking
            committedInputRef.current = input;
            startListening();
        }
    };

    const handleSend = () => {
        if (input.trim() && !isGenerating) {
            if (isListening) stopListening();
            onSendMessage(input);
            setInput('');
            committedInputRef.current = '';
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="relative w-full max-w-3xl mx-auto mt-2">
            <div
                className={`relative flex items-end p-2 pb-0 pt-0 bg-background/60 shadow-lg ring-1 rounded-2xl backdrop-blur-xl group transition-all duration-300 ${
                    isListening
                        ? 'ring-red-500/60 hover:ring-red-500/80'
                        : 'ring-white/10 hover:ring-indigo-500/40'
                }`}
            >
                <Textarea
                    value={input}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                        setInput(e.target.value);
                        if (!isListening) {
                            committedInputRef.current = e.target.value;
                        }
                    }}
                    onKeyDown={handleKeyDown}
                    placeholder={isListening ? 'Listening…' : 'Ask a question about the syllabus...'}
                    className={`min-h-[52px] max-h-[200px] w-full resize-none border-0 bg-transparent focus-visible:ring-0 px-4 py-3 text-base shadow-none outline-none focus:outline-none transition-all font-medium ${
                        isListening
                            ? 'placeholder:text-red-400/70 animate-pulse'
                            : 'placeholder:text-muted-foreground/50'
                    }`}
                    rows={1}
                />
                <div className="flex shrink-0 p-2 pl-0 gap-2 items-center mb-0.5 ml-1">
                    {/* Microphone button — hidden on unsupported browsers */}
                    {isSupported && (
                        <div className="relative">
                            {isListening && (
                                <span className="absolute inset-0 rounded-xl bg-red-500/30 animate-ping" />
                            )}
                            <Button
                                id="voice-input-btn"
                                onClick={handleMicToggle}
                                size="icon"
                                variant="ghost"
                                className={`relative h-9 w-9 rounded-xl transition-all hover:scale-105 ${
                                    isListening
                                        ? 'bg-red-600 hover:bg-red-500 text-white shadow-md shadow-red-500/30'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-white/10'
                                }`}
                                aria-label={isListening ? 'Stop recording' : 'Start voice input'}
                            >
                                {isListening ? (
                                    <MicOff className="h-4 w-4" />
                                ) : (
                                    <Mic className="h-4 w-4" />
                                )}
                            </Button>
                        </div>
                    )}

                    {isGenerating ? (
                        <Button
                            id="stop-generation-btn"
                            onClick={onStop}
                            size="icon"
                            variant="destructive"
                            className="h-9 w-9 rounded-xl shadow-md transition-transform hover:scale-105"
                        >
                            <Square className="h-4 w-4 fill-current" />
                        </Button>
                    ) : (
                        <Button
                            id="send-message-btn"
                            onClick={handleSend}
                            disabled={!input.trim()}
                            size="icon"
                            className="h-9 w-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-md transition-all hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
                        >
                            <Send className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </div>
            <p className="text-center text-xs text-muted-foreground/60 mt-3 font-medium">
                SyllabiQ AI can make mistakes. Consider checking important information.
            </p>
        </div>
    );
};

