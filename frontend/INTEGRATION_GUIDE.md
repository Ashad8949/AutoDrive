# AutoDrive Frontend Integration Guide

## What to integrate

Two files from this folder go into the frontend repo:
- `ChatWidget.tsx` — floating chat bubble (bottom-right corner, every page)
- `VoicePage.tsx` — full-screen voice assistant (dedicated `/voice` route)

---

## Step 1 — Install dependencies

```bash
npm install react react-dom
npm install --save-dev @types/react @types/react-dom
```

---

## Step 2 — Add ChatWidget to your root layout

In your root layout file (e.g. `App.tsx` or `Layout.tsx`):

```tsx
import { ChatWidget } from './ChatWidget';

export default function App() {
  return (
    <>
      {/* your existing app */}
      <YourRouter />

      {/* AutoDrive AI chat — appears on every page */}
      <ChatWidget
        apiUrl="https://autodrive-chatbot.azurewebsites.net"
        title="AutoDrive AI Assistant"
        accentColor="#1a56db"
        voiceEnabled={true}
      />
    </>
  );
}
```

---

## Step 3 — Add the Voice Assistant page

In your router (React Router v6 example):

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import VoicePage from './VoicePage';

<Routes>
  {/* your existing routes */}
  <Route path="/" element={<HomePage />} />
  <Route path="/cars" element={<CarsPage />} />

  {/* Voice assistant page */}
  <Route
    path="/voice"
    element={<VoicePage apiUrl="https://autodrive-chatbot.azurewebsites.net" />}
  />
</Routes>
```

Add a nav link to the voice page:
```tsx
<a href="/voice">🎤 Voice Assistant</a>
```

---

## Step 4 — Environment variable (recommended)

Instead of hardcoding the API URL, use an env variable:

```tsx
// .env
VITE_CHATBOT_URL=https://autodrive-chatbot.azurewebsites.net

// In your component
<ChatWidget apiUrl={import.meta.env.VITE_CHATBOT_URL} />
<VoicePage apiUrl={import.meta.env.VITE_CHATBOT_URL} />
```

---

## API Endpoints used

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /chat/stream` | POST | Streaming text chat (SSE) |
| `POST /voice/transcribe` | POST | Speech → text (Groq Whisper) |
| `GET /health` | GET | Check if chatbot is alive |

### Chat request format
```json
POST /chat/stream
{ "message": "Show me SUVs under $40k", "session_id": "uuid-string" }
```

### Voice transcription format
```
POST /voice/transcribe
Content-Type: multipart/form-data
Body: audio file (webm, mp3, wav)
Response: { "transcript": "show me SUVs under 40k" }
```

---

## ChatWidget props

| Prop | Type | Default | Description |
|---|---|---|---|
| `apiUrl` | string | `http://localhost:8002` | Chatbot backend URL |
| `title` | string | `AutoDrive AI Assistant` | Header text |
| `accentColor` | string | `#1a56db` | Brand colour |
| `voiceEnabled` | boolean | `true` | Auto-speak AI responses |

---

## Browser support

| Feature | Chrome | Edge | Firefox | Safari |
|---|---|---|---|---|
| Chat (text) | ✅ | ✅ | ✅ | ✅ |
| Mic recording | ✅ | ✅ | ✅ | ✅ |
| TTS (speak) | ✅ | ✅ | ✅ | ✅ |

> Microphone requires HTTPS in production (works on localhost without HTTPS).
