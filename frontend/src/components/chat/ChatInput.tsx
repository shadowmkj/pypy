import React, { useState, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Send, Square } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';

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

    const handleSend = () => {
        if (input.trim() && !isGenerating) {
            onSendMessage(input);
            setInput('');
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
            <div className="relative flex items-end p-2 pb-0 pt-0 bg-background/60 shadow-lg ring-1 ring-white/10 rounded-2xl backdrop-blur-xl group transition-all duration-300 hover:ring-indigo-500/40">
                <Textarea
                    value={input}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                        setInput(e.target.value)
                    }}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question about the syllabus..."
                    className="min-h-[52px] max-h-[200px] w-full resize-none border-0 bg-transparent focus-visible:ring-0 px-4 py-3 text-base shadow-none outline-none focus:outline-none placeholder:text-muted-foreground/50 transition-all font-medium"
                    rows={1}
                />
                <div className="flex shrink-0 p-2 pl-0 gap-2 items-center mb-0.5 ml-1">
                    {isGenerating ? (
                        <Button
                            onClick={onStop}
                            size="icon"
                            variant="destructive"
                            className="h-9 w-9 rounded-xl shadow-md transition-transform hover:scale-105"
                        >
                            <Square className="h-4 w-4 fill-current" />
                        </Button>
                    ) : (
                        <Button
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
