/**
 * AutoDrive Voice Assistant — Full-page standalone voice interface.
 *
 * Usage (React Router):
 *   import VoicePage from './VoicePage';
 *   <Route path="/voice" element={<VoicePage apiUrl="https://autodrive-chatbot.azurewebsites.net" />} />
 *
 * Features:
 *  - Push-to-talk big button (hold Space or click)
 *  - Groq Whisper speech-to-text
 *  - Streaming LLaMA response
 *  - Browser TTS reads AI response aloud
 *  - Conversation history shown as cards
 *  - No typing needed — fully hands-free
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface VoicePageProps {
  apiUrl?: string;
}

type Status = 'idle' | 'recording' | 'transcribing' | 'thinking' | 'speaking';

const STATUS_LABEL: Record<Status, string> = {
  idle: 'Press & hold to speak',
  recording: '🎤 Listening…',
  transcribing: '⏳ Processing speech…',
  thinking: '🤔 Thinking…',
  speaking: '🔊 Speaking… (click to stop)',
};

const STATUS_COLOR: Record<Status, string> = {
  idle: '#1a56db',
  recording: '#ef4444',
  transcribing: '#f59e0b',
  thinking: '#8b5cf6',
  speaking: '#10b981',
};

function speak(text: string, onEnd: () => void) {
  if (!('speechSynthesis' in window)) { onEnd(); return; }
  window.speechSynthesis.cancel();
  const clean = text.replace(/[*_`#]/g, '').replace(/\[ACTION:[^\]]+\]/g, '');
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 1.05;
  utt.onend = onEnd;
  utt.onerror = onEnd;
  window.speechSynthesis.speak(utt);
}

export const VoicePage: React.FC<VoicePageProps> = ({
  apiUrl = 'http://localhost:8002',
}) => {
  const [status, setStatus] = useState<Status>('idle');
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [currentTranscript, setCurrentTranscript] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // ── Space bar push-to-talk ─────────────────────────────────────────
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && status === 'idle' && e.target === document.body) {
        e.preventDefault();
        startRecording();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && status === 'recording') {
        e.preventDefault();
        stopRecording();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp); };
  }, [status]);

  // ── Recording ──────────────────────────────────────────────────────
  const startRecording = async () => {
    if (status !== 'idle') return;
    window.speechSynthesis.cancel();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        await processAudio();
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setStatus('recording');
    } catch {
      alert('Microphone permission denied. Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setStatus('transcribing');
    }
  };

  // ── Process audio → transcribe → chat → speak ─────────────────────
  const processAudio = useCallback(async () => {
    const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    let transcript = '';

    // 1. Transcribe
    try {
      const form = new FormData();
      form.append('audio', blob, 'audio.webm');
      const res = await fetch(`${apiUrl}/voice/transcribe`, { method: 'POST', body: form });
      const data = await res.json();
      transcript = data.transcript ?? '';
    } catch {
      setStatus('idle');
      return;
    }

    if (!transcript) { setStatus('idle'); return; }

    setCurrentTranscript(transcript);
    setMessages(prev => [...prev, { role: 'user', content: transcript }]);
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
    setStatus('thinking');

    // 2. Stream chat response
    let fullResponse = '';
    try {
      const res = await fetch(`${apiUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: transcript, session_id: sessionId }),
      });
      if (!res.ok || !res.body) throw new Error();
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.token) {
              fullResponse += payload.token;
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = { ...u[u.length - 1], content: u[u.length - 1].content + payload.token };
                return u;
              });
            }
          } catch { /* skip */ }
        }
      }
    } catch {
      setStatus('idle');
      return;
    }

    // 3. Speak response
    setStatus('speaking');
    setCurrentTranscript('');
    speak(fullResponse, () => setStatus('idle'));
  }, [apiUrl, sessionId]);

  const handleButtonClick = () => {
    if (status === 'idle') startRecording();
    else if (status === 'recording') stopRecording();
    else if (status === 'speaking') { window.speechSynthesis.cancel(); setStatus('idle'); }
  };

  const btnColor = STATUS_COLOR[status];
  const isActive = status === 'recording';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0f172a', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '32px 16px', fontFamily: 'system-ui, sans-serif' }}>
      <style>{`
        @keyframes ripple { 0% { transform:scale(1); opacity:0.8 } 100% { transform:scale(2.2); opacity:0 } }
        @keyframes pulse  { 0%,100% { box-shadow:0 0 0 0 ${btnColor}66 } 50% { box-shadow:0 0 0 20px ${btnColor}00 } }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <h1 style={{ color: 'white', fontSize: 28, fontWeight: 700, margin: 0 }}>🚗 AutoDrive Voice Assistant</h1>
        <p style={{ color: '#94a3b8', marginTop: 8, fontSize: 15 }}>Ask about cars, prices, or book a test drive</p>
      </div>

      {/* Conversation */}
      <div style={{ width: '100%', maxWidth: 640, flex: 1, overflowY: 'auto', marginBottom: 32, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#475569', marginTop: 60, fontSize: 15 }}>
            Press the button below or hold <kbd style={{ background: '#1e293b', padding: '2px 8px', borderRadius: 6, color: '#94a3b8', fontSize: 13 }}>Space</kbd> to start speaking
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={{
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '80%',
            backgroundColor: msg.role === 'user' ? '#1a56db' : '#1e293b',
            color: 'white', padding: '12px 16px',
            borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
            fontSize: 15, lineHeight: 1.6, whiteSpace: 'pre-wrap',
          }}>
            {msg.content || (i === messages.length - 1 && status === 'thinking' ? '▋' : '')}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Status label */}
      <p style={{ color: STATUS_COLOR[status], fontSize: 15, fontWeight: 500, marginBottom: 20, minHeight: 24 }}>
        {STATUS_LABEL[status]}
        {currentTranscript && <span style={{ color: '#94a3b8', fontWeight: 400 }}> — "{currentTranscript}"</span>}
      </p>

      {/* Big push-to-talk button */}
      <button
        onMouseDown={handleButtonClick}
        onTouchStart={e => { e.preventDefault(); handleButtonClick(); }}
        disabled={status === 'transcribing' || status === 'thinking'}
        style={{
          width: 120, height: 120, borderRadius: '50%',
          backgroundColor: btnColor, border: 'none', cursor: 'pointer',
          fontSize: 44, color: 'white',
          boxShadow: `0 8px 32px ${btnColor}66`,
          animation: isActive ? 'pulse 1s infinite' : 'none',
          transition: 'background-color 0.2s, transform 0.1s',
          transform: isActive ? 'scale(1.08)' : 'scale(1)',
          opacity: (status === 'transcribing' || status === 'thinking') ? 0.6 : 1,
          position: 'relative',
        }}
        aria-label={STATUS_LABEL[status]}
      >
        {status === 'recording' ? '⏹' : status === 'speaking' ? '🔊' : '🎤'}
      </button>

      <p style={{ color: '#475569', fontSize: 13, marginTop: 16 }}>
        {status === 'idle' ? 'Click once to start, click again to stop' : ''}
        {status === 'recording' ? 'Click to stop recording' : ''}
        {status === 'speaking' ? 'Click to stop speaking' : ''}
      </p>
    </div>
  );
};

export default VoicePage;
