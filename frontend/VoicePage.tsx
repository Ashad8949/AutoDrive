/**
 * AutoDrive Voice Assistant — Full-page voice interface.
 * Styled to match autodriveai.duckdns.org — dark slate hero, blue-600/violet gradient accents.
 *
 * Usage (React Router):
 *   <Route path="/voice" element={<VoicePage apiUrl="https://autodrive-chatbot.azurewebsites.net" />} />
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

interface Message { role: 'user' | 'assistant'; content: string; }
interface VoicePageProps { apiUrl?: string; }
type Status = 'idle' | 'recording' | 'transcribing' | 'thinking' | 'speaking';

const C = {
  blue600:    '#2563eb',
  blue400:    '#60a5fa',
  violet600:  '#7c3aed',
  violet400:  '#a78bfa',
  slate900:   '#0f172a',
  slate400:   '#94a3b8',
  slate600:   '#475569',
  slate200:   '#e2e8f0',
  white:      '#ffffff',
  green400:   '#4ade80',
  red500:     '#ef4444',
  amber500:   '#f59e0b',
  emerald500: '#10b981',
};

const STATUS_LABEL: Record<Status, string> = {
  idle:         'Hold Space or tap button to speak',
  recording:    'Listening…',
  transcribing: 'Processing speech…',
  thinking:     'Thinking…',
  speaking:     'Speaking — tap to stop',
};

const STATUS_COLOR: Record<Status, string> = {
  idle:         C.blue600,
  recording:    C.red500,
  transcribing: C.amber500,
  thinking:     C.violet600,
  speaking:     C.emerald500,
};

const STATUS_ICON: Record<Status, string> = {
  idle:         '🎤',
  recording:    '⏹',
  transcribing: '⏳',
  thinking:     '✨',
  speaking:     '🔊',
};

function speak(text: string, onEnd: () => void) {
  if (!('speechSynthesis' in window)) { onEnd(); return; }
  window.speechSynthesis.cancel();
  const clean = text.replace(/[*_`#]/g, '').replace(/\[(?:ACTION|CAR_ID):[^\]]+\]/g, '');
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 1.05;
  utt.onend = onEnd;
  utt.onerror = onEnd;
  window.speechSynthesis.speak(utt);
}

export const VoicePage: React.FC<VoicePageProps> = ({ apiUrl = 'http://localhost:8002' }) => {
  const [status, setStatus] = useState<Status>('idle');
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [currentTranscript, setCurrentTranscript] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && status === 'idle' && e.target === document.body) {
        e.preventDefault(); startRecording();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space' && status === 'recording') { e.preventDefault(); stopRecording(); }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, [status]);

  const startRecording = async () => {
    if (status !== 'idle') return;
    window.speechSynthesis.cancel();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => { stream.getTracks().forEach(t => t.stop()); await processAudio(); };
      mr.start();
      mediaRecorderRef.current = mr;
      setStatus('recording');
    } catch {
      alert('Microphone permission denied.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setStatus('transcribing');
    }
  };

  const processAudio = useCallback(async () => {
    const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    let transcript = '';
    try {
      const form = new FormData();
      form.append('audio', blob, 'audio.webm');
      const res = await fetch(`${apiUrl}/voice/transcribe`, { method: 'POST', body: form });
      transcript = (await res.json()).transcript ?? '';
    } catch { setStatus('idle'); return; }

    if (!transcript) { setStatus('idle'); return; }
    setCurrentTranscript(transcript);
    setMessages(prev => [...prev, { role: 'user', content: transcript }]);
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
    setStatus('thinking');

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
            const p = JSON.parse(line.slice(6));
            if (p.token) {
              fullResponse += p.token;
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = { ...u[u.length - 1], content: u[u.length - 1].content + p.token };
                return u;
              });
            }
          } catch { /* skip */ }
        }
      }
    } catch { setStatus('idle'); return; }

    setStatus('speaking');
    setCurrentTranscript('');
    speak(fullResponse, () => setStatus('idle'));
  }, [apiUrl, sessionId]);

  const handleButton = () => {
    if (status === 'idle') startRecording();
    else if (status === 'recording') stopRecording();
    else if (status === 'speaking') { window.speechSynthesis.cancel(); setStatus('idle'); }
  };

  const btnColor = STATUS_COLOR[status];
  const isDisabled = status === 'transcribing' || status === 'thinking';

  return (
    <div style={{
      minHeight: '100vh',
      background: `linear-gradient(135deg, ${C.slate900} 0%, #172554 50%, ${C.slate900} 100%)`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '40px 16px 120px',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    }}>
      <style>{`
        @keyframes pulse-ring  { 0%,100%{box-shadow:0 0 0 0 ${btnColor}55} 50%{box-shadow:0 0 0 24px ${btnColor}00} }
        @keyframes dot-bounce  { 0%,80%,100%{transform:translateY(0);opacity:.4} 40%{transform:translateY(-8px);opacity:1} }
        @keyframes fade-up     { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        .vmsg { animation:fade-up .3s ease; }
        .vbtn:hover:not(:disabled) { transform:scale(1.06) !important; }
      `}</style>

      {/* Back link */}
      <a href="/" style={{
        position: 'fixed', top: 20, left: 20,
        color: C.slate400, textDecoration: 'none', fontSize: 13,
        background: 'rgba(255,255,255,.06)', border: '1px solid rgba(255,255,255,.1)',
        padding: '6px 14px', borderRadius: 9999,
      }}>← Back</a>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 36, maxWidth: 520 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 14,
          background: 'rgba(37,99,235,.15)', border: '1px solid rgba(37,99,235,.3)',
          padding: '5px 14px', borderRadius: 9999,
        }}>
          <span style={{ color: C.green400, fontSize: 9 }}>●</span>
          <span style={{ color: C.blue400, fontSize: 11, fontWeight: 700, letterSpacing: '1.5px', textTransform: 'uppercase' }}>
            Voice Assistant
          </span>
        </div>
        <h1 style={{
          fontSize: 34, fontWeight: 900, letterSpacing: '-1px', margin: '0 0 10px',
          background: `linear-gradient(135deg, ${C.blue400}, ${C.violet400})`,
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        }}>AutoDrive AI</h1>
        <p style={{ color: C.slate400, fontSize: 14, margin: 0, lineHeight: 1.6 }}>
          Ask about cars, compare models, or book a test drive — completely hands-free
        </p>
      </div>

      {/* Messages */}
      <div style={{ width: '100%', maxWidth: 620, flex: 1, display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 28 }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center', padding: '40px 24px',
            background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.08)',
            borderRadius: 20,
          }}>
            <div style={{ fontSize: 40, marginBottom: 14 }}>🎙️</div>
            <p style={{ color: C.slate400, fontSize: 14, margin: '0 0 16px', lineHeight: 1.7 }}>
              Press the button below or hold{' '}
              <kbd style={{
                background: 'rgba(255,255,255,.08)', border: '1px solid rgba(255,255,255,.12)',
                padding: '2px 8px', borderRadius: 6, color: C.slate200, fontSize: 12,
              }}>Space</kbd>{' '}
              to speak
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
              {['SUVs under ₹20 lakh', 'Best EV cars', 'Book a test drive'].map(s => (
                <span key={s} style={{
                  padding: '4px 12px', borderRadius: 9999, fontSize: 12,
                  background: 'rgba(37,99,235,.12)', border: '1px solid rgba(37,99,235,.25)',
                  color: C.blue400,
                }}>{s}</span>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className="vmsg" style={{
            display: 'flex',
            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
            gap: 10, alignItems: 'flex-end',
          }}>
            {msg.role === 'assistant' && (
              <div style={{
                width: 32, height: 32, borderRadius: 10, flexShrink: 0,
                background: `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`,
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
              }}>🤖</div>
            )}
            <div style={{
              maxWidth: '78%', padding: '12px 16px', fontSize: 14, lineHeight: 1.65,
              whiteSpace: 'pre-wrap', color: C.white,
              borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              background: msg.role === 'user'
                ? `linear-gradient(135deg, ${C.blue600}, ${C.violet600})`
                : 'rgba(255,255,255,.07)',
              border: msg.role === 'assistant' ? '1px solid rgba(255,255,255,.1)' : 'none',
              backdropFilter: 'blur(8px)',
            }}>
              {msg.content || (i === messages.length - 1 && status === 'thinking' ? (
                <span style={{ display: 'inline-flex', gap: 4 }}>
                  {[0, 160, 320].map(d => (
                    <span key={d} style={{
                      width: 7, height: 7, borderRadius: '50%', background: C.slate400,
                      display: 'inline-block',
                      animation: `dot-bounce 1.2s ease-in-out ${d}ms infinite`,
                    }} />
                  ))}
                </span>
              ) : '')}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Status pill */}
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '8px 20px', borderRadius: 9999,
          background: `${btnColor}18`, border: `1px solid ${btnColor}40`,
          transition: 'all .3s',
        }}>
          <span style={{ fontSize: 14 }}>{STATUS_ICON[status]}</span>
          <span style={{ color: btnColor, fontSize: 13, fontWeight: 600 }}>{STATUS_LABEL[status]}</span>
          {currentTranscript && (
            <span style={{ color: C.slate400, fontSize: 13 }}>— "{currentTranscript}"</span>
          )}
        </div>
      </div>

      {/* Big button */}
      <button
        className="vbtn"
        onMouseDown={handleButton}
        onTouchStart={e => { e.preventDefault(); handleButton(); }}
        disabled={isDisabled}
        aria-label={STATUS_LABEL[status]}
        style={{
          width: 96, height: 96, borderRadius: '50%', border: 'none',
          cursor: isDisabled ? 'not-allowed' : 'pointer', fontSize: 34, color: C.white,
          background: isDisabled
            ? 'rgba(255,255,255,.08)'
            : `linear-gradient(135deg, ${btnColor}, ${status === 'idle' ? C.violet600 : btnColor})`,
          boxShadow: isDisabled ? 'none' : `0 8px 32px ${btnColor}50`,
          animation: status === 'recording' ? 'pulse-ring 1.2s ease-in-out infinite' : 'none',
          transition: 'background .25s, box-shadow .25s, transform .15s',
          transform: status === 'recording' ? 'scale(1.08)' : 'scale(1)',
          opacity: isDisabled ? 0.4 : 1,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        {STATUS_ICON[status]}
      </button>

      <p style={{ color: C.slate600, fontSize: 12, marginTop: 12, textAlign: 'center' }}>
        {status === 'idle' && 'Click once to start · click again to stop'}
        {status === 'recording' && 'Click to stop recording'}
        {status === 'speaking' && 'Click to stop speaking'}
      </p>
    </div>
  );
};

export default VoicePage;
