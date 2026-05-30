# 🐦‍🔥 Phoenix AI: Unified API Integration Guide

This guide explains how to connect any external technology—such as **Flutter**, **.NET**, **React Native**, or **Unity**—to the Phoenix AI Framework using our ready-to-use FastAPI server.

---

##  The Quick-Start API Server

We provide a pre-configured FastAPI server in `phoenix_api.py`. This server exposes a single `/chat` endpoint that handles text, images (Vision), and audio (STT) automatically.

### Running the Server
```bash
python phoenix_api.py
```
The server will start at `http://localhost:8000`. You can access the auto-generated documentation at `http://localhost:8000/docs`.

---

## 📱 Flutter Integration

In Flutter, use the `http` package to send multi-part requests to the Phoenix API.

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<void> sendToPhoenix(String prompt, String? sessionId) async {
  var request = http.MultipartRequest(
    'POST', 
    Uri.parse('http://YOUR_SERVER_IP:8000/chat')
  );
  
  request.fields['prompt'] = prompt;
  if (sessionId != null) request.fields['session_id'] = sessionId;

  // Optional: Add image
  // request.files.add(await http.MultipartFile.fromPath('image', 'path/to/img.jpg'));

  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);
  
  if (response.statusCode == 200) {
    var data = json.decode(response.body);
    print("Phoenix: ${data['reply']}");
  }
}
```

---

## 💻 .NET / C# Integration

Use `HttpClient` and `MultipartFormDataContent` to interact with the API from a Windows Desktop or ASP.NET app.

```csharp
using System.Net.Http;

public async Task<string> AskPhoenix(string prompt, string sessionId = null)
{
    using var client = new HttpClient();
    using var content = new MultipartFormDataContent();

    content.Add(new StringContent(prompt), "prompt");
    if (sessionId != null) content.Add(new StringContent(sessionId), "session_id");

    var response = await client.PostAsync("http://YOUR_SERVER_IP:8000/chat", content);
    var json = await response.Content.ReadAsStringAsync();
    
    // Parse JSON and return reply...
    return json;
}
```

---

## 🌐 Generic HTTP (cURL)

Test your endpoint instantly from the terminal:

```bash
# Simple Text Chat
curl -X POST http://localhost:8000/chat \
     -F "prompt=Hello Phoenix!"

# Vision Chat (with image)
curl -X POST http://localhost:8000/chat \
     -F "prompt=What is in this image?" \
     -F "image=@/path/to/your/image.jpg"
```

---

## 🛠️ API Reference: `/chat`

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `string` | **Yes** | The user message or question. |
| `session_id`| `string` | No | Persistent session ID for memory context. |
| `image` | `file` | No | Image file for Vision (VLM) processing. |
| `audio` | `file` | No | Audio file for transcription and response. |

### Response Schema
```json
{
  "status": "success",
  "reply": "The response text from Phoenix AI",
  "session_id": "user_session_123"
}
```

---

## 🔒 Security & CORS
The provided `phoenix_api.py` includes **CORS** middleware enabled with `allow_origins=["*"]`. This is essential for browser-based apps and Flutter web to communicate with the backend without security blocks.
