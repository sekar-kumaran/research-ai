# Frontend: app.js

## File Path: `frontend/app.js`
## Status: Active / Stable

## Description
This is the single JavaScript file powering the entire frontend UI. It is heavily modularized without using a build step or bundler. It relies exclusively on Vanilla JS and modern DOM APIs (e.g., `fetch` and Server-Sent Events).

## Core Responsibilities & Components

### 1. State Management
- Maintains a global `state` object holding:
  - `conversationId`: UUID for the current active chat session.
  - `history`: Local cache of past conversations.
  - `theme`: UI color scheme.
  - `topK`: Slider setting for retrieval limits.
  - `streaming`: Boolean lock preventing duplicate sends while waiting for a response.

### 2. DOM Selection
- Uses a shorthand `$` function to select elements by ID.
- Defines constants for all critical UI elements at the top of the file.

### 3. API Communication (`callApi`)
- A wrapper around `fetch` that automatically attaches the JWT token (if `auth_required` is implemented) and handles JSON parsing and error throwing.

### 4. Real-time Streaming (`/chat/stream`)
- **Function**: `sendMessage(query, sessionId)`
- Opens an SSE (Server-Sent Event) stream.
- Parses `data: { ... }` chunks and applies markdown rendering dynamically.
- Intercepts backend events:
  - `start`: Records the `conversation_id`.
  - `sources`: Buffers cited papers.
  - `done`: Finalizes the UI bubble, computes latency, and saves to history.

### 5. History Management
- **Function**: `addToHistory(title)`
  - Pushes a new entry to the `state.history` array, or bumps an existing `conversationId` to the top.
  - Persists to `localStorage` under `rai-history`.
- **Function**: `loadConversation(id, title)`
  - Fetches past turns from `/conversations/{id}` and repopulates the UI.
  - Handles server-side expiration (404) gracefully by starting a new chat.

### 6. Document Upload & arXiv Loader
- POSTs files to `/chat/upload` or triggers `/chat/load-arxiv`.
- Tracks uploaded documents in `state.loadedSessions`.
