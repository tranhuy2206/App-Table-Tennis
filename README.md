---
title: TTATool Backend
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# TTATool Backend

FastAPI backend for the Table Tennis Assessment Toolkit.

## Required Secret

Set this value in the Hugging Face Space settings:

```text
GOOGLE_API_KEY=your_google_api_key
```

## Useful Endpoints

- `/health`
- `/docs`
- `/api/chatbot/status`
- `/api/chatbot/init`
- `/api/video/search`
