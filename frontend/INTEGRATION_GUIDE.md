# AutoDrive AI Chatbot — Integration Guide

**Chatbot backend URL:** `https://autodrive-chatbot.azurewebsites.net`

---

## Files to copy

From `https://github.com/Ashad8949/AutoDrive/tree/master/frontend`:

| File | Copy to |
|---|---|
| `ChatWidget.tsx` | `src/components/ChatWidget.tsx` |
| `VoicePage.tsx` | `src/app/voice/page.tsx` (or `src/pages/voice.tsx`) |

---

## Step 1 — Copy the files

```bash
# In your project root
curl -o src/components/ChatWidget.tsx \
  https://raw.githubusercontent.com/Ashad8949/AutoDrive/master/frontend/ChatWidget.tsx

curl -o src/components/VoicePage.tsx \
  https://raw.githubusercontent.com/Ashad8949/AutoDrive/master/frontend/VoicePage.tsx
```

---

## Step 2 — Add ChatWidget to your root layout

Since your project is **Next.js 14 with App Router**, add it to `src/app/layout.tsx`:

```tsx
// src/app/layout.tsx
import { ChatWidget } from '@/components/ChatWidget';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}

        {/* AutoDrive AI — appears on every page */}
        <ChatWidget
          apiUrl="https://autodrive-chatbot.azurewebsites.net"
          title="AutoDrive AI"
          voiceEnabled={true}
          onCarMentioned={(carId) => {
            // Optional: navigate to car details when bot mentions a car
            window.location.href = `/cars/${carId}`;
          }}
        />
      </body>
    </html>
  );
}
```

> If you use **Pages Router** (`pages/`), add it to `pages/_app.tsx` instead:
> ```tsx
> import { ChatWidget } from '../components/ChatWidget';
> export default function App({ Component, pageProps }) {
>   return <>
>     <Component {...pageProps} />
>     <ChatWidget apiUrl="https://autodrive-chatbot.azurewebsites.net" />
>   </>;
> }
> ```

---

## Step 3 — Add the Voice Assistant page

**App Router** — create `src/app/voice/page.tsx`:

```tsx
'use client';
import { VoicePage } from '@/components/VoicePage';

export default function Voice() {
  return <VoicePage apiUrl="https://autodrive-chatbot.azurewebsites.net" />;
}
```

**Pages Router** — create `pages/voice.tsx`:

```tsx
import { VoicePage } from '../components/VoicePage';
export default function Voice() {
  return <VoicePage apiUrl="https://autodrive-chatbot.azurewebsites.net" />;
}
```

Add a nav link anywhere in your navbar:
```tsx
<a href="/voice">🎤 Voice Assistant</a>
```

---

## Step 4 — Handle the `onCarMentioned` callback (deep-link feature)

When the bot mentions a car (e.g. "Hyundai Creta"), it emits a `car_id`. Wire it up for one-click navigation:

```tsx
import { useRouter } from 'next/navigation'; // App Router
// OR: import { useRouter } from 'next/router'; // Pages Router

function Layout({ children }) {
  const router = useRouter();

  return (
    <>
      {children}
      <ChatWidget
        apiUrl="https://autodrive-chatbot.azurewebsites.net"
        onCarMentioned={(carId) => router.push(`/cars/${carId}`)}
      />
    </>
  );
}
```

---

## Step 5 — Environment variable (recommended)

```bash
# .env.local
NEXT_PUBLIC_CHATBOT_URL=https://autodrive-chatbot.azurewebsites.net
```

```tsx
<ChatWidget apiUrl={process.env.NEXT_PUBLIC_CHATBOT_URL} />
<VoicePage apiUrl={process.env.NEXT_PUBLIC_CHATBOT_URL} />
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /chat/stream` | POST | Streaming chat (SSE) |
| `POST /voice/transcribe` | POST | Speech → text (Groq Whisper) |
| `POST /inventory/refresh` | POST | Force-refresh car inventory |
| `GET /health` | GET | Health check |

### SSE Event types your frontend receives:

```
data: {"token": "The Hyundai "}     ← stream token to UI
data: {"token": "Creta "}
data: {"car_id": "2"}               ← bot mentioned car ID 2 → deep-link to /cars/2
data: {"action": "BOOK_TEST_DRIVE 2"} ← show booking widget
data: {"done": true}                ← stream complete
```

---

## ChatWidget props

| Prop | Type | Default | Description |
|---|---|---|---|
| `apiUrl` | string | `http://localhost:8002` | Chatbot backend URL |
| `title` | string | `AutoDrive AI` | Header text |
| `voiceEnabled` | boolean | `true` | Speak AI responses aloud |
| `onCarMentioned` | `(id: string) => void` | — | Called when bot names a car |

---

## CORS

The backend already allows all origins (`*`). No CORS config needed on your side.

## Browser support

| Feature | Chrome | Edge | Firefox | Safari |
|---|---|---|---|---|
| Chat | ✅ | ✅ | ✅ | ✅ |
| Mic recording | ✅ | ✅ | ✅ | ✅ |
| TTS (speak) | ✅ | ✅ | ✅ | ✅ |

> Microphone requires HTTPS in production (localhost works without HTTPS).
